"""
instrumented_tabpfn.py
--------------------------

  1. Capture per-head attention weights for both attention modules at every layer.
  2. Zero individual head outputs (per layer / per attn_type) for ablation.

Architecture note (TabPFN v2):
  - model.blocks  is a ModuleList of TabPFNBlock
  - Each block has:
      per_sample_attention_between_features  (AlongRowAttention)   — feature attn
      per_column_attention_between_cells     (AlongColumnAttention) — datapoint attn


Usage
-----
    from tabpfn import TabPFNClassifier
    clf = TabPFNClassifier(n_estimators=1, device='cpu')
    clf.fit(X_train, y_train)

    inst = InstrumentedTabPFN(clf.model_)
    inst.enable_capture()
    proba = clf.predict_proba(X_test)
    caps = inst.get_captured()
    # caps['feature'][layer_idx]    -> tensor (Br, H, C, C)
    # caps['datapoint'][layer_idx]  -> tensor (Bc, H, R, N)

    inst.set_masks({(layer_idx, head_idx, 'datapoint'): True})
    proba_ablated = clf.predict_proba(X_test)

    inst.restore()   # undo all patches
"""

from __future__ import annotations

import math
import types
from typing import Dict, List, Optional, Tuple

import torch

MaskKey = Tuple[int, int, str]



class _InstrState:
    def __init__(self):
        self.capture: bool = False
        self.captured: Dict[str, Dict[int, torch.Tensor]] = {
            "feature": {}, "datapoint": {},
        }
        # masks_by_layer[(layer_idx, attn_type)] = set of head indices to zero
        self.masks_by_layer: Dict[Tuple[int, str], set] = {}


# helpers

def _manual_attn(
    q: torch.Tensor,   # (B, S, H, D)
    k: torch.Tensor,   # (B, T, H, D)  T=S or T=N (keys only)
    v: torch.Tensor,   # (B, T, H, D)
    masked_heads: Optional[set] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        out     : (B, S, H, D)
        weights : (B, H, S, T)
    """
    D = q.shape[3]
    scale = 1.0 / math.sqrt(D)

    # (B, H, S, D) x (B, H, D, T) -> (B, H, S, T)
    logits = torch.einsum("bshd,bthd->bhst", q, k) * scale
    weights = torch.softmax(logits, dim=-1)

    # (B, H, S, T) x (B, T, H, D) -> (B, S, H, D)
    out = torch.einsum("bhst,bthd->bshd", weights, v)

    if masked_heads:
        for h in masked_heads:
            out[:, :, h, :] = 0.0

    return out, weights



# Capturing forward attention

def _patched_row_forward_factory(state: _InstrState, layer_idx: int):
    """AlongRowAttention.forward replacement — captures feature attention."""

    def forward(self, x_BrSE: torch.Tensor) -> torch.Tensor:
        Br, C, _ = x_BrSE.shape
        q = self.q_projection(x_BrSE).view(Br, C, -1, self.head_dim)  # (Br, C, H, D)
        k = self.k_projection(x_BrSE).view(Br, C, -1, self.head_dim)
        v = self.v_projection(x_BrSE).view(Br, C, -1, self.head_dim)

        masked = state.masks_by_layer.get((layer_idx, "feature"))
        out, weights = _manual_attn(q, k, v, masked_heads=masked)
        # weights: (Br, H, C, C)

        if state.capture:
            state.captured["feature"][layer_idx] = weights.detach().cpu()

        out_flat = out.reshape(Br, C, self.head_dim * self.num_heads)
        return self.out_projection(out_flat)

    return forward


def _patched_col_forward_factory(state: _InstrState, layer_idx: int):
    """AlongColumnAttention.forward replacement — captures datapoint attention.

    Matches the original's multi-query attention: when single_eval_pos is set
    and != R, test rows use only head-0 of K/V (all query heads attend to the
    same key/value head).
    """

    def forward(self, x_BcRE: torch.Tensor, single_eval_pos=None) -> torch.Tensor:
        Bc, R, _ = x_BcRE.shape
        N = R if single_eval_pos is None else single_eval_pos

        q = self.q_projection(x_BcRE).view(Bc, R, -1, self.head_dim)          # (Bc, R, H, D)
        k = self.k_projection(x_BcRE[:, :N]).view(Bc, N, -1, self.head_dim)   # (Bc, N, H, D)
        v = self.v_projection(x_BcRE[:, :N]).view(Bc, N, -1, self.head_dim)

        masked = state.masks_by_layer.get((layer_idx, "datapoint"))
        H = q.shape[2]

        if single_eval_pos is None or single_eval_pos == R:
            out, weights = _manual_attn(q, k, v, masked_heads=masked)
        else:
            out_train, w_train = _manual_attn(
                q[:, :N], k, v, masked_heads=masked
            )
            k_h0 = k[:, :, :1, :].expand(-1, -1, H, -1)
            v_h0 = v[:, :, :1, :].expand(-1, -1, H, -1)
            out_test, w_test = _manual_attn(
                q[:, N:], k_h0, v_h0, masked_heads=masked
            )
            out = torch.cat([out_train, out_test], dim=1)
            weights = torch.cat([w_train, w_test], dim=2)

        if state.capture:
            state.captured["datapoint"][layer_idx] = weights.detach().cpu()

        out_flat = out.reshape(Bc, R, self.head_dim * self.num_heads)
        return self.out_projection(out_flat)

    return forward


# Main wrapper

class InstrumentedTabPFN:
    """Monkey-patches a fitted TabPFN v2 model for attention capture + ablation.

    Patching is in-place on the model; subsequent clf.predict_proba() calls
    will run through the instrumented attention.  Call restore() to undo.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.state = _InstrState()
        self._originals: List[Tuple[object, str, object]] = []  # (module, attr, orig_fn)
        self._patch()

    def _patch(self):
        for layer_idx, block in enumerate(self.model.blocks):
            for _, attn_mod, factory in (
                ("feature",   block.per_sample_attention_between_features,
                 _patched_row_forward_factory),
                ("datapoint", block.per_column_attention_between_cells,
                 _patched_col_forward_factory),
            ):
                orig = attn_mod.forward
                self._originals.append((attn_mod, "forward", orig))
                new_fn = factory(self.state, layer_idx)
                attn_mod.forward = types.MethodType(new_fn, attn_mod)

    def restore(self):
        for mod, attr, orig in self._originals:
            setattr(mod, attr, orig)
        self._originals = []

   

    def enable_capture(self):
        self.state.capture = True

    def disable_capture(self):
        self.state.capture = False

    def clear_capture(self):
        self.state.captured = {"feature": {}, "datapoint": {}}

    def get_captured(self) -> Dict[str, Dict[int, torch.Tensor]]:
        """Returns {attn_type: {layer_idx: tensor}}."""
        return self.state.captured

    
    def set_masks(self, masked_heads: Dict[MaskKey, bool]):
        """Keys: (layer_idx, head_idx, attn_type). Set to True to zero that head."""
        self.state.masks_by_layer = {}
        for (l, h, t), flag in masked_heads.items():
            if not flag:
                continue
            self.state.masks_by_layer.setdefault((l, t), set()).add(h)

    def clear_masks(self):
        self.state.masks_by_layer = {}

    
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.restore()
