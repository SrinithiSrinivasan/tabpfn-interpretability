"""
permutation_test.py
-------------------
Permutation test: is the observed performance drop (from freezing OBSERVED_FREEZE)
more extreme than random chance?

Null distribution = N_RUNS random sets of the same size drawn from the sampling pool.

Usage:
  python permutation_test.py --model tabpfn
  python permutation_test.py --model tabpfn --pool non_aligned
  python permutation_test.py --model nano   --checkpoint path/to/model.pt
"""

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from backends import get_backend
from find_heads import load_selected_heads
from helpers import build_splits, scores as compute_scores

N_RUNS = 50
METRIC = "roc_auc"


def eval_freeze(backend, splits, freeze_list):
    mask  = {h: True for h in freeze_list}
    drops = {}
    for ds, (X_tr, X_te, y_tr, y_te) in splits.items():
        backend.fit(X_tr, y_tr)
        base = compute_scores(y_te, backend.predict_proba(X_te))
        froz = compute_scores(y_te, backend.predict_frozen(X_te, mask))
        drops[ds] = {m: froz[m] - base[m] for m in base}
    return drops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      choices=["nano", "tabpfn"], required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--pool",       choices=["non_aligned", "full"], default="non_aligned",
                        help="non_aligned: exclude OBSERVED_FREEZE from null pool (default); "
                             "full: sample from all heads")
    args = parser.parse_args()

    out_dir = ROOT / "results" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = ROOT / "figures" / args.model
    fig_dir.mkdir(parents=True, exist_ok=True)

    OBSERVED_FREEZE = load_selected_heads(out_dir)
    print(f"  Loaded {len(OBSERVED_FREEZE)} heads from selected_heads.json")

    random.seed(0)
    kw      = {} if args.model == "tabpfn" else {"checkpoint": args.checkpoint}
    backend = get_backend(args.model, **kw)
    splits  = build_splits(backend.supported_datasets)

    # fit once to get architecture
    first = next(iter(splits.values()))
    backend.fit(first[0], first[2])
    nl, nh = backend.n_layers, backend.n_heads

    # Build per-type pools to match the type composition of OBSERVED_FREEZE.
    # e.g. 5 feature + 10 datapoint → sample 5 from feature pool, 10 from datapoint pool.
    observed_set    = set(OBSERVED_FREEZE)
    type_counts     = {}
    for _, _, t in OBSERVED_FREEZE:
        type_counts[t] = type_counts.get(t, 0) + 1

    pools = {}
    for t, count in type_counts.items():
        universe = [(l, h, t) for l in range(nl) for h in range(nh)]
        pools[t] = ([x for x in universe if x not in observed_set]
                    if args.pool == "non_aligned" else universe)

    n_freeze = len(OBSERVED_FREEZE)
    datasets = list(splits.keys())
    pool_summary = "  +  ".join(f"{count} {t} from {len(pools[t])}"
                                 for t, count in type_counts.items())
    print(f"  Sampling: {pool_summary}  |  {N_RUNS} runs")

    # observed run
    print(f"\nRunning OBSERVED set ({n_freeze} heads)...")
    obs_drops = eval_freeze(backend, splits, OBSERVED_FREEZE)
    for ds in datasets:
        print(f"  {ds:<20} {METRIC}: {obs_drops[ds][METRIC]:+.4f}")

    # null distribution
    print(f"\nRunning {N_RUNS} null ablations...")
    null_runs = []
    for i in range(N_RUNS):
        freeze = [x for t, count in type_counts.items()
                  for x in random.sample(pools[t], count)]
        null_runs.append({"freeze": freeze,
                          "drops":  eval_freeze(backend, splits, freeze)})
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{N_RUNS}")

    # save
    (out_dir / "permutation_null.json").write_text(
        json.dumps({"observed": obs_drops, "null_runs": null_runs,
                    "config": {"n_runs": N_RUNS, "n_freeze": n_freeze,
                               "metric": METRIC, "pool": args.pool}}, indent=2))

    # stats + plot
    print(f"\n{'='*60}")
    print(f"  PERMUTATION TEST  —  model={args.model}  metric={METRIC}  pool={args.pool}")
    print(f"{'='*60}")

    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5), squeeze=False)
    pool_label = "non-aligned pool" if args.pool == "non_aligned" else "full pool"
    fig.suptitle(f"Permutation test ({args.model}) — {N_RUNS}×{n_freeze} heads "
                 f"({pool_label})\nMetric: {METRIC}  |  red = observed", fontsize=11)

    for col, ds in enumerate(datasets):
        null_arr = np.array([r["drops"][ds][METRIC] for r in null_runs])
        obs_val  = obs_drops[ds][METRIC]
        z        = (obs_val - null_arr.mean()) / (null_arr.std() + 1e-9)
        p_val    = float((null_arr <= obs_val).mean())

        print(f"\n  {ds}")
        print(f"    observed : {obs_val:+.4f}")
        print(f"    null     : {null_arr.mean():+.4f} ± {null_arr.std():.4f}")
        print(f"    z-score  : {z:+.2f}")
        print(f"    p-value  : {p_val:.3f}  ({'✓ sig.' if p_val < 0.05 else 'n.s.'})")

        ax = axes[0, col]
        ax.hist(null_arr, bins=15, color="#4a90d9", alpha=0.75, edgecolor="white")
        ax.axvline(obs_val,          color="red",   lw=2,   label=f"obs={obs_val:+.3f}")
        ax.axvline(null_arr.mean(),  color="black", lw=1.2, ls="--",
                   label=f"null={null_arr.mean():+.3f}")
        ax.set_xlabel(f"{METRIC} drop", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_title(f"{ds}\nz={z:+.2f}  p={p_val:.3f}  "
                     f"({'✓' if p_val < 0.05 else 'n.s.'})", fontsize=9)
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_fig = fig_dir / "permutation_test.png"
    fig.savefig(out_fig, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure → {out_fig}")


if __name__ == "__main__":
    main()
