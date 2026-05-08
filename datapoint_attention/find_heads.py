"""
find_heads.py
-------------
Read saved attention caches, compute class-alignment scores, and save all heads
above the threshold to results/<model>/selected_heads.json.

freeze_eval.py and permutation_test.py read from that JSON automatically.

Usage:
  python find_heads.py --model tabpfn --thresh 0.65
  python find_heads.py --model nano   --thresh 0.65 --checkpoint path/to/model.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from backends import get_backend
from class_align import compute_scores_from_cache
from helpers import build_splits

HEADS_FILE = "selected_heads.json"


def save_selected_heads(high_heads: dict, threshold: float, out_dir: Path) -> list:
    """Deduplicate across datasets, sort, save to JSON. Returns the union list."""
    seen, union = set(), []
    for heads in high_heads.values():
        for lh in heads:
            if lh not in seen:
                seen.add(lh)
                union.append(lh)
    union.sort()

    out = {
        "threshold": threshold,
        "by_dataset": {ds: [[l, h, "datapoint"] for l, h in heads]
                       for ds, heads in high_heads.items()},
        "union": [[l, h, "datapoint"] for l, h in union],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / HEADS_FILE).write_text(json.dumps(out, indent=2))
    print(f"\n  Saved {len(union)} heads → {out_dir / HEADS_FILE}")
    return union


def load_selected_heads(out_dir: Path) -> list:
    """Load union head list from JSON. Returns list of (layer, head, attn_type)."""
    path = out_dir / HEADS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run find_heads.py first")
    data = json.loads(path.read_text())
    return [tuple(h) for h in data["union"]]


def print_results(all_scores, nl, nh, threshold):
    high_heads = {}
    for ds, ca in all_scores.items():
        above = [(li, hi) for li in range(nl) for hi in range(nh)
                 if not np.isnan(ca[li, hi]) and ca[li, hi] > threshold]
        high_heads[ds] = above
        print(f"\n{'='*55}")
        print(f"  {ds}  —  class-align > {threshold:.2f}")
        print(f"{'='*55}")
        if not above:
            print("  (none)")
        else:
            print(f"  {'Layer':<8} {'Head':<8} {'Score':>8}")
            print(f"  {'-'*24}")
            for li, hi in above:
                print(f"  L{li:<7} H{hi:<7} {ca[li, hi]:>8.3f}")
        print(f"  Total: {len(above)} / {nl * nh}")
    return high_heads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      choices=["nano", "tabpfn"], required=True)
    parser.add_argument("--thresh",     type=float, default=0.65)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--cache_dir",  default=None)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir) if args.cache_dir else ROOT / "results" / args.model
    kw        = {} if args.model == "tabpfn" else {"checkpoint": args.checkpoint}
    backend   = get_backend(args.model, **kw)
    splits    = build_splits(backend.supported_datasets)

    all_scores, nl, nh = compute_scores_from_cache(backend, splits, cache_dir)
    high_heads = print_results(all_scores, nl, nh, args.thresh)
    save_selected_heads(high_heads, args.thresh, cache_dir)


if __name__ == "__main__":
    main()
