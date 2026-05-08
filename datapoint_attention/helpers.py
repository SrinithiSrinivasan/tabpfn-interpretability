"""
helpers.py
----------
Shared utilities: dataset loading, metrics, eval, plotting, freeze evaluation.
Imported by find_heads.py, permutation_test.py, and run.py.
"""

from __future__ import annotations

import json
import ssl
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

TEST_SIZE = 0.5
SEED      = 42

NANO_DATASETS   = ["breast_cancer", "wine_binary", "digits_binary"]
TABPFN_DATASETS = ["breast_cancer", "wine", "digits"]


# ── Datasets ──────────────────────────────────────────────────────────────────

def _fetch_openml_safe(name, version=1):
    from sklearn.datasets import fetch_openml
    try:
        return fetch_openml(name, version=version, as_frame=False, parser="liac-arff")
    except Exception:
        orig = ssl._create_default_https_context
        ssl._create_default_https_context = ssl._create_unverified_context
        try:
            return fetch_openml(name, version=version, as_frame=False, parser="liac-arff")
        finally:
            ssl._create_default_https_context = orig


def load_dataset(name: str):
    """Returns (X: float32, y: int)."""
    if name == "breast_cancer":
        X, y = load_breast_cancer(return_X_y=True)
    elif name == "wine":
        X, y = load_wine(return_X_y=True)
    elif name == "digits":
        X, y = load_digits(return_X_y=True)
    elif name == "wine_binary":
        X, y = load_wine(return_X_y=True)
        mask = y < 2        # classes 0 vs 1 only (NanoTabPFN 2-output limit)
        X, y = X[mask], y[mask]
    elif name == "digits_binary":
        X, y = load_digits(n_class=2, return_X_y=True)
    else:
        raise ValueError(f"Unknown dataset: {name!r}")
    return X.astype(np.float32), y.astype(int)


def build_splits(dataset_names: List[str], test_size: float = TEST_SIZE,
                 seed: int = SEED) -> dict:
    """Returns {name: (X_tr, X_te, y_tr, y_te)}, skipping failed loads."""
    splits = {}
    for name in dataset_names:
        try:
            X, y = load_dataset(name)
            splits[name] = train_test_split(X, y, test_size=test_size,
                                            random_state=seed, stratify=y)
        except Exception as e:
            print(f"  [skip] {name}: {e}")
    return splits


# ── Metrics ───────────────────────────────────────────────────────────────────

def roc_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    if proba.shape[1] == 2:
        return float(roc_auc_score(y_true, proba[:, 1]))
    return float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))


def scores(y_te: np.ndarray, proba: np.ndarray) -> dict:
    pred = proba.argmax(axis=1)
    return {
        "roc_auc":      roc_auc(y_te, proba),
        "acc":          float(accuracy_score(y_te, pred)),
        "balanced_acc": float(balanced_accuracy_score(y_te, pred)),
    }


# ── Eval (fit + capture caches + save scores) ─────────────────────────────────

def run_eval(backend, out_dir: Path) -> None:
    """Fit backend on each dataset, save attention caches and scores."""
    out_dir.mkdir(parents=True, exist_ok=True)
    splits     = build_splits(backend.supported_datasets)
    scores_out = {}
    nl = nh = None

    for ds, (X_tr, X_te, y_tr, y_te) in splits.items():
        print(f"\n{ds}: {len(X_tr)} train / {len(X_te)} test / {X_tr.shape[1]} feat")
        backend.fit(X_tr, y_tr)

        cache = backend.capture(X_te)
        proba = backend.predict_proba(X_te)
        s     = scores(y_te, proba)
        s.update({"n_train": len(X_tr), "n_test": len(X_te),
                  "n_features": int(X_tr.shape[1]),
                  "n_classes":  int(len(np.unique(y_tr)))})
        scores_out[ds] = s
        print(f"  roc_auc={s['roc_auc']:.4f}  acc={s['acc']:.4f}")

        if nl is None:
            nl, nh = backend.n_layers, backend.n_heads

        # Average over batch dimension (axis 0) before saving.
        # Raw shape: feature (Br, H, C, C), datapoint (Bc, H, R, N)
        # Saved shape: (H, C, C) and (H, R, N) — reduces 50GB to ~500MB.
        arrays = {f"{t}_attn_L{l}": np.ascontiguousarray(arr.mean(axis=0))
                  for t in ("feature", "datapoint")
                  for l, arr in cache[t].items()}
        np.savez(str(out_dir / f"attention_cache_{ds}.npz"), **arrays)
        print(f"  cache saved ({len(arrays)} arrays)")

    (out_dir / "config.json").write_text(
        json.dumps({"num_layers": nl, "num_heads": nh, "model": backend.name}, indent=2))
    (out_dir / "scores.json").write_text(json.dumps(scores_out, indent=2))
    print(f"\nEval done → {out_dir}")


# ── Freeze eval ───────────────────────────────────────────────────────────────

def run_freeze_eval(backend, freeze_list: list, out_dir: Path) -> dict:
    """
    Fit backend on each dataset, compare baseline vs frozen performance.
    Saves results + the freeze list to out_dir/freeze_results.json.
    Returns {dataset: {metric: {baseline, frozen, drop}}}.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    splits  = build_splits(backend.supported_datasets)
    mask    = {h: True for h in freeze_list}
    results = {}

    print(f"{'Dataset':<18} {'Metric':<12} {'Baseline':>10} {'Frozen':>10} {'Drop':>10}")
    print("-" * 65)

    for ds, (X_tr, X_te, y_tr, y_te) in splits.items():
        backend.fit(X_tr, y_tr)
        base = scores(y_te, backend.predict_proba(X_te))
        froz = scores(y_te, backend.predict_frozen(X_te, mask))

        ds_result = {}
        heads_str = ", ".join(f"L{l}H{h}_{t[0]}" for l, h, t in freeze_list)
        for metric in ("roc_auc", "acc", "balanced_acc"):
            drop = froz[metric] - base[metric]
            tag  = "▼" if drop < -0.005 else ("▲" if drop > 0.005 else " ")
            suffix = f"  {heads_str}" if metric == "roc_auc" else ""
            print(f"  {ds:<16} {metric:<12} {base[metric]:>10.4f} "
                  f"{froz[metric]:>10.4f} {drop:>+10.4f}{tag}{suffix}")
            ds_result[metric] = {"baseline": base[metric],
                                 "frozen":   froz[metric],
                                 "drop":     drop}
        results[ds] = ds_result
        print()

    out = {
        "freeze_list": [[l, h, t] for l, h, t in freeze_list],
        "results":     results,
    }
    (out_dir / "freeze_results.json").write_text(json.dumps(out, indent=2))
    print(f"Freeze results → {out_dir / 'freeze_results.json'}")
    return results


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_class_align(all_scores: dict, nl: int, nh: int,
                     out_dir: Path, model_name: str) -> None:
    for ds, ca in all_scores.items():
        fig, ax = plt.subplots(figsize=(max(5, nh * 1.5), max(4, nl * 0.45)))
        im = ax.imshow(ca, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn")
        plt.colorbar(im, ax=ax, label="Class-alignment score")
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_xticks(range(nh))
        ax.set_yticks(range(nl))
        ax.set_title(f"{model_name} — {ds} — class-alignment", fontsize=10)
        for li in range(nl):
            for hi in range(nh):
                v = ca[li, hi]
                if not np.isnan(v):
                    ax.text(hi, li, f"{v:.2f}", ha="center", va="center", fontsize=7,
                            color="black" if 0.3 < v < 0.7 else "white")
        plt.tight_layout()
        out = out_dir / f"class_align_{ds}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  {out.name}")


def plot_attn_pages(npz: dict, nl: int, nh: int, attn_type: str,
                    ds: str, out_dir: Path) -> None:
    page_size = 6
    for page_start in range(0, nl, page_size):
        page_end = min(page_start + page_size, nl)
        n_rows   = page_end - page_start
        fig, axes = plt.subplots(n_rows, nh,
                                 figsize=(nh * 2.8, n_rows * 2.2), squeeze=False)
        fig.suptitle(f"{attn_type} attention — {ds} "
                     f"(L{page_start}–L{page_end-1})", fontsize=10)
        for row_i, li in enumerate(range(page_start, page_end)):
            key = f"{attn_type}_attn_L{li}"
            arr = npz.get(key)   # (H, R, N) — already batch-averaged
            for hi in range(nh):
                ax = axes[row_i, hi]
                if arr is not None:
                    ax.imshow(arr[hi], aspect="auto", cmap="Blues")
                ax.set_title(f"L{li}H{hi}", fontsize=8)
                ax.axis("off")
        plt.tight_layout()
        suffix = f"_p{page_start // page_size}" if nl > page_size else ""
        out = out_dir / f"{attn_type}_attn_{ds}{suffix}.png"
        fig.savefig(out, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  {out.name}")


def run_plot(backend, cache_dir: Path, out_dir: Path) -> None:
    from class_align import compute_scores_from_cache
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = build_splits(backend.supported_datasets)

    print("Computing class-alignment scores...")
    all_scores, nl, nh = compute_scores_from_cache(backend, splits, cache_dir)

    print("Plotting class-alignment heatmaps:")
    plot_class_align(all_scores, nl, nh, out_dir, backend.name)

    for ds in splits:
        npz_path = cache_dir / f"attention_cache_{ds}.npz"
        if not npz_path.exists():
            continue
        npz = dict(np.load(npz_path))
        print(f"\nAttention heatmaps — {ds}:")
        for attn_type in ("feature", "datapoint"):
            plot_attn_pages(npz, nl, nh, attn_type, ds, out_dir)

    print(f"\nFigures → {out_dir}")
