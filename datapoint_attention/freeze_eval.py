"""
freeze_eval.py
--------------
Freeze the heads selected by find_heads.py (stored in results/<model>/selected_heads.json)
and measure performance drop per dataset.

Run find_heads.py first to generate the JSON, then:
  python freeze_eval.py --model tabpfn
  python freeze_eval.py --model nano --checkpoint path/to/model.pt
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from backends import get_backend
from find_heads import load_selected_heads
from helpers import run_freeze_eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      choices=["nano", "tabpfn"], required=True)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    out_dir = ROOT / "results" / args.model
    heads   = load_selected_heads(out_dir)
    print(f"Loaded {len(heads)} heads from selected_heads.json")

    kw      = {} if args.model == "tabpfn" else {"checkpoint": args.checkpoint}
    backend = get_backend(args.model, **kw)
    run_freeze_eval(backend, heads, out_dir)
