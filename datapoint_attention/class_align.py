"""Class-alignment score computation shared across both models."""

import json
import numpy as np
from pathlib import Path

CUMUL_THRESH = 0.80



def class_align_score(da_mean: np.ndarray, y_tr: np.ndarray, y_te: np.ndarray) -> float:
    """
    da_mean : (n_test, n_train) — attention weights from each test example to train set.
    k is chosen per test example as the minimum number of train examples
    whose cumulative attention weight reaches 80% of the total.
    Returns mean fraction of top-k train examples sharing the test label.
    """
    row_scores = []
    for ti in range(len(y_te)):
        row        = da_mean[ti]
        sorted_idx = np.argsort(row)[::-1]
        cumsum     = np.cumsum(row[sorted_idx])
        k          = max(int(np.searchsorted(cumsum, CUMUL_THRESH) + 1), 1)
        top_idx    = sorted_idx[:k]
        row_scores.append((y_tr[top_idx] == y_te[ti]).sum() / k)
    return float(np.mean(row_scores))


def compute_scores_from_cache(backend, splits: dict, cache_dir: Path):
    """
    Load saved .npz caches and compute class-alignment for every (layer, head).
    Returns (all_scores, n_layers, n_heads)
      all_scores : {dataset: np.ndarray(nl, nh)}
    """
    cfg = json.loads((cache_dir / "config.json").read_text())
    nl, nh = cfg["num_layers"], cfg["num_heads"]

    all_scores = {}
    for ds, split in splits.items():
        npz_path = cache_dir / f"attention_cache_{ds}.npz"
        if not npz_path.exists():
            print(f"  [skip] no cache for {ds}")
            continue

        cache   = dict(np.load(npz_path))
        X_tr, _, y_tr, y_te = split
        n_train = len(X_tr)
        ca      = np.full((nl, nh), np.nan)

        for li in range(nl):
            key = f"datapoint_attn_L{li}"
            if key not in cache:
                continue
            da_mean = backend.extract_test_attn(cache[key], n_train)  # (H, n_test, n_train)
            for hi in range(nh):
                ca[li, hi] = class_align_score(da_mean[hi], y_tr, y_te)

        all_scores[ds] = ca

    return all_scores, nl, nh
