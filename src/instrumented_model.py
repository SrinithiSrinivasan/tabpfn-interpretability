"""
instrumented_model.py
---------------------
A  wrapper around model.py that exposes:

  1. Per-head attention weights for BOTH attention types at every layer
     - feature attention:   A ∈ ℝ^{C×C}  (columns attend to each other)
     - datapoint attention: A ∈ ℝ^{N×N}  (rows attend to each other)

  2. Layer-wise activations (output of each TransformerEncoderLayer)

  3. Head zeroing for ablation (E1): pass masked_heads={(layer, head, attn_type): True}
     to zero a specific head's contribution before the residual add.

"""

from collections import OrderedDict
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from model import NanoTabPFNModel, FeatureEncoder, TargetEncoder, Decoder


# ──────────────────────────────────────────────────────────────────────────────
# Instrumented transformer layer
# ──────────────────────────────────────────────────────────────────────────────

class InstrumentedTransformerLayer(nn.Module):
    """
    Mirrors TransformerEncoderLayer exactly but:
      - returns per-head attention weights for both attention operations
      - supports zeroing individual head outputs before the residual add
    """

    def __init__(self, source_layer):
        """Copy all submodules from an existing TransformerEncoderLayer."""
        super().__init__()
        self.disable_datapoint_attention = getattr(source_layer, 'disable_datapoint_attention', False)
        if not self.disable_datapoint_attention:
            self.self_attention_between_datapoints = source_layer.self_attention_between_datapoints
        self.self_attention_between_features   = source_layer.self_attention_between_features
        self.linear1 = source_layer.linear1
        self.linear2 = source_layer.linear2
        self.norm1   = source_layer.norm1
        self.norm2   = source_layer.norm2
        self.norm3   = source_layer.norm3

    def forward(
        self,
        src: torch.Tensor,
        train_test_split_index: int,
        masked_heads_feat: Optional[Dict[int, bool]] = None,
        masked_heads_dp:   Optional[Dict[int, bool]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        src : (B, rows, cols, E)
        train_test_split_index : int
        masked_heads_feat : {head_idx: True} — zero these feature-attn heads
        masked_heads_dp   : {head_idx: True} — zero these datapoint-attn heads

        Returns
        -------
        src          : (B, rows, cols, E)  updated tensor
        feat_attn_w  : (B*rows, n_heads, cols, cols)
        dp_attn_w    : (B*cols, n_heads, n_train+n_test, n_train)
                       first n_train rows are train→train weights,
                       last  n_test  rows are test→train weights
        """
        batch_size, rows_size, col_size, embedding_size = src.shape
        n_heads = self.self_attention_between_features.num_heads

        #  feature attention 
        src_f = src.reshape(batch_size * rows_size, col_size, embedding_size)
        feat_out, feat_attn_w = self.self_attention_between_features(
            src_f, src_f, src_f,
            need_weights=True,
            average_attn_weights=False,   # keep per-head: (B*rows, n_heads, C, C)
        )
        # head zeroing for feature attention
        if masked_heads_feat:
            feat_out = self._zero_heads(feat_out, feat_attn_w, src_f, masked_heads_feat,
                                        self.self_attention_between_features)
        src_f = feat_out + src_f
        src   = src_f.reshape(batch_size, rows_size, col_size, embedding_size)
        src   = self.norm1(src)

        #  datapoint attention 
        if not self.disable_datapoint_attention:
            src = src.transpose(1, 2)                                    # (B, C, rows, E)
            src = src.reshape(batch_size * col_size, rows_size, embedding_size)

            train_src = src[:, :train_test_split_index]                  # (B*C, n_train, E)
            test_src  = src[:, train_test_split_index:]                  # (B*C, n_test,  E)

            # train attends to itself
            dp_out_train, dp_attn_train = self.self_attention_between_datapoints(
                train_src, train_src, train_src,
                need_weights=True,
                average_attn_weights=False,   # (B*C, n_heads, n_train, n_train)
            )
            # test attends to training data
            dp_out_test, dp_attn_test = self.self_attention_between_datapoints(
                test_src, train_src, train_src,
                need_weights=True,
                average_attn_weights=False,   # (B*C, n_heads, n_test, n_train)
            )

            # head zeroing for datapoint attention
            if masked_heads_dp:
                dp_out_train = self._zero_heads(dp_out_train, dp_attn_train, train_src,
                                                masked_heads_dp,
                                                self.self_attention_between_datapoints)
                dp_out_test  = self._zero_heads(dp_out_test,  dp_attn_test,  test_src,
                                                masked_heads_dp,
                                                self.self_attention_between_datapoints,
                                                key_src=train_src)

            dp_out = torch.cat([dp_out_train, dp_out_test], dim=1)       # (B*C, rows, E)
            src = dp_out + src
            src = src.reshape(batch_size, col_size, rows_size, embedding_size)
            src = src.transpose(2, 1)                                    # (B, rows, C, E)
        src = self.norm2(src)

        #  MLP 
        src = self.linear2(F.gelu(self.linear1(src))) + src
        src = self.norm3(src)

        # concatenate train+test attn weights along query dimension for easy indexing
        if not self.disable_datapoint_attention:
            dp_attn_w = torch.cat([dp_attn_train, dp_attn_test], dim=2)  # (B*C, n_heads, rows, n_train)
        else:
            dp_attn_w = None

        return src, feat_attn_w, dp_attn_w

    # helper

    @staticmethod
    def _zero_heads(
        attn_out: torch.Tensor,
        attn_w:   torch.Tensor,
        query_src: torch.Tensor,
        masked_heads: Dict[int, bool],
        mha_module,
        key_src: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Re-derives the per-head contributions and zeroes the specified ones.

        attn_out   : (batch, seq_q, embed)  — full projected output from MHA
        attn_w     : (batch, n_heads, seq_q, seq_k)  — softmax weights per head
        query_src  : (batch, seq_q, embed)  — the query input
        key_src    : (batch, seq_k, embed)  — if None, assumed == query_src (self-attn)
        mha_module : the nn.MultiheadAttention instance (for weight access)

        
        """
        if key_src is None:
            key_src = query_src

        batch, seq_q, embed = query_src.shape
        n_heads   = mha_module.num_heads
        head_dim  = embed // n_heads

        # Extract in_proj weights: PyTorch packs Q/K/V together
        in_proj_weight = mha_module.in_proj_weight   # (3*embed, embed)
        in_proj_bias   = mha_module.in_proj_bias     # (3*embed,)

        # V projection: last third of in_proj
        V_w = in_proj_weight[2 * embed:, :]           # (embed, embed)
        V_b = in_proj_bias[2 * embed:]                # (embed,)

        V = F.linear(key_src, V_w, V_b)               # (batch, seq_k, embed)
        V = V.reshape(batch, -1, n_heads, head_dim).transpose(1, 2)   # (batch, n_heads, seq_k, head_dim)

        # attn_w: (batch, n_heads, seq_q, seq_k)
        # weighted sum over keys
        head_out = torch.einsum("bhqk,bhkd->bhqd", attn_w, V)         # (batch, n_heads, seq_q, head_dim)

        # zero requested heads
        for h in masked_heads:
            head_out[:, h, :, :] = 0.0

        # merge heads and out-project
        merged = head_out.transpose(1, 2).reshape(batch, seq_q, embed)  # (batch, seq_q, embed)
        out_proj_weight = mha_module.out_proj.weight   # (embed, embed)
        out_proj_bias   = mha_module.out_proj.bias     # (embed,)
        return F.linear(merged, out_proj_weight, out_proj_bias)


# ──────────────────────────────────────────────────────────────────────────────
# Instrumented full model
# ──────────────────────────────────────────────────────────────────────────────

class InstrumentedNanoTabPFN(nn.Module):
    """
    Wraps a trained NanoTabPFNModel, sharing all weights, and adds
    forward_with_cache() for mechanistic analysis.
    """

    def __init__(self, base_model: NanoTabPFNModel):
        super().__init__()
        self.feature_encoder = base_model.feature_encoder
        self.target_encoder  = base_model.target_encoder
        self.decoder         = base_model.decoder
        self.transformer_blocks = nn.ModuleList(
            [InstrumentedTransformerLayer(blk) for blk in base_model.transformer_blocks]
        )
        self.num_layers = len(self.transformer_blocks)

    def forward(self, src, train_test_split_index):
        """Standard forward — identical to NanoTabPFNModel, no capture overhead."""
        logits, _ = self.forward_with_cache(
            src[0], src[1], train_test_split_index=train_test_split_index
        )
        return logits

    def forward_with_cache(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        train_test_split_index: int,
        masked_heads: Optional[Dict[Tuple[int, int, str], bool]] = None,
    ) -> Tuple[torch.Tensor, OrderedDict]:
        """
        Run a full forward pass and return both logits and a cache dict.

        Parameters
        ----------
        x : (B, rows, features)  float tensor
        y : (B, n_train)         float tensor (labels for training rows)
        train_test_split_index : int
        masked_heads : optional dict mapping (layer_idx, head_idx, attn_type) → True
            where attn_type is "feature" or "datapoint"
            e.g. {(0, 2, "datapoint"): True} zeros layer-0 datapoint head-2

        Returns
        -------
        logits : (B, n_test, num_classes)
        cache  : OrderedDict with keys:
            "feature_attn"   : list[L] of (B*rows, n_heads, C, C)
            "datapoint_attn" : list[L] of (B*cols, n_heads, rows, n_train)
            "activations"    : list[L] of (B, rows, cols, embed)
                               index 0 = output of layer 0, etc.
        """
        masked_heads = masked_heads or {}

        # encoders
        if len(y.shape) < len(x.shape):
            y = y.unsqueeze(-1)

        x_enc = self.feature_encoder(x, train_test_split_index)
        y_enc = self.target_encoder(y, x_enc.shape[1])
        src   = torch.cat([x_enc, y_enc], dim=2)    # (B, rows, cols, E)

        # transformer blocks 
        feat_attn_list = []
        dp_attn_list   = []
        act_list       = []

        for layer_idx, block in enumerate(self.transformer_blocks):
            # collect masks for this layer
            mh_feat = {h: True for (l, h, t), v in masked_heads.items()
                       if l == layer_idx and t == "feature"}
            mh_dp   = {h: True for (l, h, t), v in masked_heads.items()
                       if l == layer_idx and t == "datapoint"}

            src, feat_w, dp_w = block(
                src,
                train_test_split_index=train_test_split_index,
                masked_heads_feat=mh_feat or None,
                masked_heads_dp=mh_dp or None,
            )

            feat_attn_list.append(feat_w)
            dp_attn_list.append(dp_w)
            act_list.append(src.detach().clone())

        #  decode 
        output = src[:, train_test_split_index:, -1, :]   # (B, n_test, E)
        logits = self.decoder(output)

        cache = OrderedDict([
            ("feature_attn",   feat_attn_list),
            ("datapoint_attn", dp_attn_list),
            ("activations",    act_list),
        ])
        return logits, cache

    
    def as_classifier(self, device=None):
        """Return an InstrumentedClassifier wrapping this model."""
        if device is None:
            device = next(self.parameters()).device
        return InstrumentedClassifier(self, device)


class InstrumentedClassifier:
    """
    sklearn-like wrapper for InstrumentedNanoTabPFN.
    Mirrors NanoTabPFNClassifier but also exposes the last forward cache.
    """

    def __init__(self, model: InstrumentedNanoTabPFN, device):
        self.model  = model.to(device)
        self.device = device
        self.last_cache: Optional[OrderedDict] = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.X_train    = X_train
        self.y_train    = y_train
        self.num_classes = int(max(y_train)) + 1

    def predict_proba(
        self,
        X_test: np.ndarray,
        masked_heads: Optional[Dict[Tuple[int, int, str], bool]] = None,
    ) -> np.ndarray:
        x = np.concatenate((self.X_train, X_test))
        y = self.y_train
        with torch.no_grad():
            x_t = torch.from_numpy(x).unsqueeze(0).float().to(self.device)
            y_t = torch.from_numpy(y).unsqueeze(0).float().to(self.device)
            logits, cache = self.model.forward_with_cache(
                x_t, y_t,
                train_test_split_index=len(self.X_train),
                masked_heads=masked_heads,
            )
            self.last_cache = cache
            logits = logits.squeeze(0)[:, :self.num_classes]
            return F.softmax(logits, dim=1).cpu().numpy()

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        return self.predict_proba(X_test, **kwargs).argmax(axis=1)
