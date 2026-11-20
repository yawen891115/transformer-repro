"""Average the weights of the last N epoch checkpoints (optional; often +0.3-0.5 BLEU).

    python scripts/average_checkpoints.py --expdir logs/iwslt14_deen --last 10
    python generate.py --expdir logs/iwslt14_deen --ckpt checkpoint_avg10.pt

Report averaged results as such ("average of the last 10 epochs"); the fairseq README
number uses checkpoint_best.pt without averaging.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--expdir", required=True)
    parser.add_argument("--last", type=int, default=10)
    args = parser.parse_args()

    expdir = Path(args.expdir)
    epochs = sorted(int(m.group(1)) for p in expdir.glob("checkpoint*.pt")
                    if (m := re.fullmatch(r"checkpoint(\d+)\.pt", p.name)))
    chosen = epochs[-args.last:]
    if len(chosen) < args.last:
        print(f"warning: only {len(chosen)} epoch checkpoints available")
    avg, last = None, None
    for e in chosen:
        last = torch.load(expdir / f"checkpoint{e}.pt", map_location="cpu", weights_only=False)
        state = last["model"]
        if avg is None:
            avg = {k: v.clone().double() for k, v in state.items()}
        else:
            for k in avg:
                avg[k] += state[k].double()
    model_state = {k: (v / len(chosen)).to(last["model"][k].dtype) for k, v in avg.items()}
    out = expdir / f"checkpoint_avg{len(chosen)}.pt"
    torch.save({**last, "model": model_state, "averaged_epochs": chosen}, out)
    print(f"averaged epochs {chosen} -> {out}")


if __name__ == "__main__":
    main()
