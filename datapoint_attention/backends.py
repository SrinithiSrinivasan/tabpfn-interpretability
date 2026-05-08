"""
backends.py
-----------
Unified ModelBackend interface for NanoTabPFN and TabPFN v2.

Both backends expose the same four operations:
  fit(X_train, y_train)
  predict_proba(X_test)                         → probabilities (no capture)
  capture(X_test)                               → {"feature": {L: tensor}, "datapoint": {L: tensor}}
  predict_frozen(X_test, {(L,H,type): True})    → probabilities with heads zeroed

Usage:
  from backends import get_backend
  b = get_backend("tabpfn")
  b = get_backend("nano", checkpoint="path/to/model.pt")
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

MaskKey = Tuple[int, int, str]


class ModelBackend(ABC):

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def n_layers(self) -> int: ...

    @property
    @abstractmethod
    def n_heads(self) -> int: ...

    @property
    @abstractmethod
    def supported_datasets(self) -> List[str]: ...

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None: ...

    @abstractmethod
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def capture(self, X_test: np.ndarray) -> Dict[str, Dict[int, np.ndarray]]:
        """Returns {"feature": {layer: array}, "datapoint": {layer: array}}."""
        ...

    @abstractmethod
    def predict_frozen(self, X_test: np.ndarray,
                       freeze: Dict[MaskKey, bool]) -> np.ndarray: ...

    @abstractmethod
    def extract_test_attn(self, da_layer: np.ndarray, n_train: int) -> np.ndarray:
        """Return (H, n_test, n_train) mean-over-batch test→train attention slice."""
        ...


# ---------------------------------------------------------------------------
# NanoTabPFN backend
# ---------------------------------------------------------------------------

class NanoBackend(ModelBackend):
    """
    Wraps InstrumentedNanoTabPFN + InstrumentedClassifier.
    Architecture is read from a config.json sitting next to the checkpoint,
    or falls back to the standard 3L-4H config.
    """

    SUPPORTED = ["breast_cancer", "wine_binary", "digits_binary"]

    def __init__(self, checkpoint: str, device: str = "cpu"):
        self.checkpoint = Path(checkpoint)
        self.device = device
        self._clf = None
        self._nl: Optional[int] = None
        self._nh: Optional[int] = None

    def _build_clf(self):
        from model import NanoTabPFNModel
        from instrumented_model_nanotabpfn import InstrumentedNanoTabPFN

        cfg_path = self.checkpoint.parent / "config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            emb  = cfg.get("embedding_size", 96)
            nh   = cfg.get("num_heads", 4)
            mlp  = cfg.get("mlp_hidden_size", 192)
            nl   = cfg.get("num_layers", 3)
            nout = cfg.get("num_outputs", 2)
        else:
            emb, nh, mlp, nl, nout = 96, 4, 192, 3, 2

        base = NanoTabPFNModel(embedding_size=emb, num_attention_heads=nh,
                               mlp_hidden_size=mlp, num_layers=nl, num_outputs=nout)
        state = torch.load(self.checkpoint, map_location="cpu")
        base.load_state_dict(state)
        inst = InstrumentedNanoTabPFN(base)
        inst.eval()
        self._nl = inst.num_layers
        self._nh = inst.transformer_blocks[0].self_attention_between_features.num_heads
        return inst.as_classifier(device=self.device)

    @property
    def name(self) -> str:
        return "nano"

    @property
    def n_layers(self) -> int:
        if self._nl is None:
            raise RuntimeError("Call fit() first")
        return self._nl

    @property
    def n_heads(self) -> int:
        if self._nh is None:
            raise RuntimeError("Call fit() first")
        return self._nh

    @property
    def supported_datasets(self) -> List[str]:
        return self.SUPPORTED

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        if self._clf is None:
            self._clf = self._build_clf()
        self._clf.fit(X_train, y_train)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        return self._clf.predict_proba(X_test)

    def capture(self, X_test: np.ndarray) -> Dict[str, Dict[int, np.ndarray]]:
        self._clf.predict_proba(X_test)
        raw = self._clf.last_cache
        return {
            "feature":   {i: t.numpy() for i, t in enumerate(raw["feature_attn"])},
            "datapoint": {i: t.numpy() for i, t in enumerate(raw["datapoint_attn"])},
        }

    def predict_frozen(self, X_test: np.ndarray,
                       freeze: Dict[MaskKey, bool]) -> np.ndarray:
        return self._clf.predict_proba(X_test, masked_heads=freeze)

    def extract_test_attn(self, da_layer: np.ndarray, n_train: int) -> np.ndarray:
        # da_layer: (H, n_train+n_test, n_train) — batch-averaged before saving
        return da_layer[:, n_train:, :]         # (H, n_test, n_train)


# ---------------------------------------------------------------------------
# TabPFN v2 backend
# ---------------------------------------------------------------------------

class TabPFNBackend(ModelBackend):
    """
    Wraps TabPFNClassifier + InstrumentedTabPFN (monkey-patched flash attention).
    Re-instruments after every fit() because clf.model_ is recreated.
    """

    SUPPORTED = ["breast_cancer", "wine", "digits"]
    SEED = 42

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._clf = None
        self._inst = None
        self._nl: Optional[int] = None
        self._nh: Optional[int] = None

    @property
    def name(self) -> str:
        return "tabpfn"

    @property
    def n_layers(self) -> int:
        if self._nl is None:
            raise RuntimeError("Call fit() first")
        return self._nl

    @property
    def n_heads(self) -> int:
        if self._nh is None:
            raise RuntimeError("Call fit() first")
        return self._nh

    @property
    def supported_datasets(self) -> List[str]:
        return self.SUPPORTED

    def _ensure_clf(self):
        if self._clf is None:
            from tabpfn import TabPFNClassifier
            self._clf = TabPFNClassifier(
                n_estimators=1, device=self.device,
                random_state=self.SEED, ignore_pretraining_limits=True,
            )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._ensure_clf()
        self._clf.fit(X_train, y_train)
        from instrumented_model_tabpfn import InstrumentedTabPFN
        if self._inst is not None:
            self._inst.restore()
        self._inst = InstrumentedTabPFN(self._clf.model_)
        blocks = self._clf.model_.blocks
        self._nl = len(blocks)
        self._nh = blocks[0].per_sample_attention_between_features.num_heads

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        return self._clf.predict_proba(X_test)

    def capture(self, X_test: np.ndarray) -> Dict[str, Dict[int, np.ndarray]]:
        self._inst.enable_capture()
        self._inst.clear_capture()
        self._clf.predict_proba(X_test)
        self._inst.disable_capture()
        raw = self._inst.get_captured()
        return {
            "feature":   {l: t.numpy() for l, t in raw["feature"].items()},
            "datapoint": {l: t.numpy() for l, t in raw["datapoint"].items()},
        }

    def predict_frozen(self, X_test: np.ndarray,
                       freeze: Dict[MaskKey, bool]) -> np.ndarray:
        self._inst.set_masks(freeze)
        out = self._clf.predict_proba(X_test)
        self._inst.clear_masks()
        return out

    def extract_test_attn(self, da_layer: np.ndarray, n_train: int) -> np.ndarray:
        # da_layer: (H, R, N) — batch-averaged before saving; N includes thinking rows
        thinking_rows = da_layer.shape[2] - n_train
        q_start       = thinking_rows + n_train
        return da_layer[:, q_start:, thinking_rows:]        # (H, n_test, n_train)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_backend(name: str, **kwargs) -> ModelBackend:
    if name == "nano":
        return NanoBackend(**kwargs)
    if name == "tabpfn":
        return TabPFNBackend(**kwargs)
    raise ValueError(f"Unknown backend '{name}'. Choose 'nano' or 'tabpfn'.")
