"""
run.py
------
Launcher for the unified attention experiment pipeline.

Steps:
  eval     — fit model, capture attention caches, save scores
  find     — identify high class-alignment heads from caches
  freeze   — freeze selected heads and measure performance drop
  permute  — permutation test (statistical significance)
  plot     — generate heatmap + class-alignment figures

Usage:
  python run.py --model tabpfn --step all
  python run.py --model tabpfn --step eval find
  python run.py --model nano --checkpoint path/to/model.pt --step all
  python run.py --model tabpfn --step permute --pool non_aligned
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# ★ EDIT THESE defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL      = "tabpfn"
DEFAULT_CHECKPOINT = None          
DEFAULT_THRESH     = 0.65          
DEFAULT_POOL       = "non_aligned"        
AUTO_UPDATE_FREEZE = False         
# ---------------------------------------------------------------------------

HERE       = Path(__file__).parent
PYTHON     = sys.executable
STEP_ORDER = ["eval", "find", "freeze", "permute", "plot"]


def _backend(args):
    sys.path.insert(0, str(HERE))
    from backends import get_backend
    kw = {} if args.model == "tabpfn" else {"checkpoint": args.checkpoint}
    return get_backend(args.model, **kw)


def step_eval(args):
    from helpers import run_eval
    b       = _backend(args)
    out_dir = HERE / "results" / args.model
    run_eval(b, out_dir)


def step_plot(args):
    from helpers import run_plot
    b         = _backend(args)
    cache_dir = HERE / "results" / args.model
    out_dir   = HERE / "figures" / args.model
    run_plot(b, cache_dir, out_dir)


def step_freeze(args):
    from helpers import run_freeze_eval
    from find_heads import load_selected_heads
    b       = _backend(args)
    out_dir = HERE / "results" / args.model
    heads   = load_selected_heads(out_dir)
    print(f"  Loaded {len(heads)} heads from selected_heads.json")
    run_freeze_eval(b, heads, out_dir)


def step_subprocess(script_name, args, extra_args=None):
    """Run find_heads.py or permutation_test.py as subprocess."""
    cmd = [PYTHON, str(HERE / script_name),
           "--model", args.model]
    if args.checkpoint:
        cmd += ["--checkpoint", args.checkpoint]
    if extra_args:
        cmd += extra_args
    print(f"\n{'='*60}\n  {script_name}  →  {' '.join(cmd)}\n{'='*60}\n")
    result = subprocess.run(cmd, cwd=HERE)
    if result.returncode != 0:
        sys.exit(result.returncode)


STEP_FNS = {
    "eval":    lambda args: step_eval(args),
    "plot":    lambda args: step_plot(args),
    "freeze":  lambda args: step_freeze(args),
    "find":    lambda args: step_subprocess(
        "find_heads.py", args,
        ["--thresh", str(args.thresh)] + (["--update"] if args.update else [])),
    "permute": lambda args: step_subprocess(
        "permutation_test.py", args,
        ["--pool", args.pool]),
}


def main():
    parser = argparse.ArgumentParser(description="Attention experiment launcher")
    parser.add_argument("--model",      choices=["nano", "tabpfn"], default=DEFAULT_MODEL)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--step",       nargs="+", default=["all"],
                        choices=STEP_ORDER + ["all"])
    parser.add_argument("--thresh",     type=float, default=DEFAULT_THRESH)
    parser.add_argument("--pool",       choices=["non_aligned", "full"], default=DEFAULT_POOL)
    parser.add_argument("--update",     action="store_true", default=AUTO_UPDATE_FREEZE)
    args = parser.parse_args()

    steps = STEP_ORDER if "all" in args.step else args.step
    print(f"Model: {args.model}  |  Steps: {' → '.join(steps)}")

    for step in steps:
        print(f"\n── {step} ──")
        STEP_FNS[step](args)

    print(f"\n✓ Done — {len(steps)} step(s) completed.")


if __name__ == "__main__":
    main()
