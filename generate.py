"""Translate the test set with beam search and report BLEU (like fairseq-generate --remove-bpe).

    python generate.py --expdir logs/iwslt14_deen                       # checkpoint_best.pt
    python generate.py --expdir logs/iwslt14_deen --ckpt checkpoint_avg10.pt --sacrebleu

The main number is tokenised, lower-cased BLEU after removing BPE, which is how IWSLT'14
De->En results are usually reported (fairseq: ~34.4-35 BLEU for this model).
--sacrebleu additionally detokenises both sides with Moses and reports sacreBLEU.
Writes generate-<split>.txt with S (source), T (reference) and H (hypothesis) lines.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from transformer.bleu import corpus_bleu
from transformer.config import load_config
from transformer.data import ParallelData, collate, make_batches
from transformer.search import beam_search
from transformer.utils import build_model, load_checkpoint, load_dictionaries, pick_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--expdir", required=True)
    parser.add_argument("--ckpt", default="checkpoint_best.pt")
    parser.add_argument("--split", default="test", choices=["test", "valid", "train"])
    parser.add_argument("--beam", type=int, default=5)
    parser.add_argument("--lenpen", type=float, default=1.0)
    parser.add_argument("--max-len-a", type=float, default=0.0)
    parser.add_argument("--max-len-b", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128, help="sentences per batch")
    parser.add_argument("--sacrebleu", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    expdir = Path(args.expdir)
    cfg = load_config(str(expdir / "config.yaml"))
    device = pick_device(args.device)
    src_dict, tgt_dict = load_dictionaries(cfg)
    model = build_model(cfg, src_dict, tgt_dict).to(device).eval()
    ckpt_path = Path(args.ckpt) if Path(args.ckpt).exists() else expdir / args.ckpt
    if not ckpt_path.exists() and args.ckpt == "checkpoint_best.pt":
        ckpt_path = expdir / "checkpoint_last.pt"      # e.g. training stopped before BLEU started
        print(f"no checkpoint_best.pt yet; using {ckpt_path.name}")
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")
    ckpt = load_checkpoint(ckpt_path, model)
    data = ParallelData.load(f"{cfg.data_bin}/{args.split}.pt")

    hyps, refs, srcs = [""] * len(data), [""] * len(data), [""] * len(data)
    t0 = time.time()
    batches = make_batches(data, max_tokens=10 ** 9, shuffle=False, max_sentences=args.batch_size)
    for idx in tqdm(batches, desc=f"beam={args.beam}"):
        b = collate(data, idx, tgt_dict.pad, tgt_dict.bos)
        outs = beam_search(model, b["src_tokens"].to(device), args.beam, tgt_dict.bos,
                           tgt_dict.eos, tgt_dict.pad, args.max_len_a, args.max_len_b, args.lenpen)
        for i, out in zip(idx, outs):
            hyps[i] = tgt_dict.decode(out, remove_bpe=True)
            refs[i] = tgt_dict.decode(data.tgt[i].tolist(), remove_bpe=True, escape_unk=True)
            srcs[i] = src_dict.decode(data.src[i].tolist(), remove_bpe=True)
    seconds = time.time() - t0

    out_txt = expdir / f"generate-{args.split}.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        for i in range(len(data)):
            f.write(f"S-{i}\t{srcs[i]}\nT-{i}\t{refs[i]}\nH-{i}\t{hyps[i]}\n")

    s = corpus_bleu(hyps, refs)
    prec = "/".join(f"{p:.1f}" for p in s["precisions"])
    line = (f"BLEU4 = {s['bleu']:.2f}, {prec} (BP={s['bp']:.3f}, "
            f"ratio={s['hyp_len'] / s['ref_len']:.3f}, syslen={s['hyp_len']}, "
            f"reflen={s['ref_len']})")
    print(f"checkpoint {ckpt_path.name} (epoch {ckpt['epoch']}), {args.split}: {len(data)} "
          f"sentences in {seconds:.0f}s")
    print(line)
    result = {"checkpoint": str(ckpt_path), "split": args.split, "beam": args.beam,
              "tokenized_bleu": s["bleu"], "detail": line}

    if args.sacrebleu:
        import sacrebleu
        from sacremoses import MosesDetokenizer
        detok = MosesDetokenizer(lang=cfg.tgt_lang)
        d_h = [detok.detokenize(h.split()) for h in hyps]
        d_r = [detok.detokenize(r.split()) for r in refs]
        sb = sacrebleu.corpus_bleu(d_h, [d_r])
        print(f"sacreBLEU (detokenised, data already lower-cased): {sb.score:.2f}  {sb}")
        result["sacrebleu"] = sb.score

    (expdir / f"bleu-{args.split}.json").write_text(json.dumps(result, indent=2))
    print(f"wrote {out_txt}")


if __name__ == "__main__":
    main()
