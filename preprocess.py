"""Build the vocabularies and binarise the tokenised + BPE text (like fairseq-preprocess).

    python preprocess.py --textdir data/iwslt14.tokenized.de-en \
                         --destdir data-bin/iwslt14.tokenized.de-en

Reads {train,valid,test}.{de,en} and writes dict.de.txt, dict.en.txt and {train,valid,test}.pt.
As in the fairseq IWSLT'14 example, source and target get separate dictionaries
(pass --joined-dictionary to share one, which also allows share_all_embeddings).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from transformer.data import Dictionary, ParallelData


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--textdir", required=True)
    parser.add_argument("--destdir", required=True)
    parser.add_argument("--source-lang", default="de")
    parser.add_argument("--target-lang", default="en")
    parser.add_argument("--joined-dictionary", action="store_true")
    args = parser.parse_args()

    text, dest = Path(args.textdir), Path(args.destdir)
    src, tgt = args.source_lang, args.target_lang
    dest.mkdir(parents=True, exist_ok=True)

    if args.joined_dictionary:
        src_dict = tgt_dict = Dictionary.build([text / f"train.{src}", text / f"train.{tgt}"])
    else:
        src_dict = Dictionary.build([text / f"train.{src}"])
        tgt_dict = Dictionary.build([text / f"train.{tgt}"])
    src_dict.save(dest / f"dict.{src}.txt")
    tgt_dict.save(dest / f"dict.{tgt}.txt")
    print(f"[{src}] dictionary: {len(src_dict)} types   [{tgt}] dictionary: {len(tgt_dict)} types")

    for split in ("train", "valid", "test"):
        data = ParallelData.from_text(text / f"{split}.{src}", text / f"{split}.{tgt}",
                                      src_dict, tgt_dict)
        data.save(dest / f"{split}.pt")
        n_src, n_tgt = int(data.src_lens.sum()), int(data.tgt_lens.sum())
        unk_src = sum(int((s == src_dict.unk).sum()) for s in data.src)
        unk_tgt = sum(int((t == tgt_dict.unk).sum()) for t in data.tgt)
        print(f"{split:5s}: {len(data):7d} sentences, {n_src:8d} {src} tokens "
              f"({100 * unk_src / n_src:.2f}% <unk>), {n_tgt:8d} {tgt} tokens "
              f"({100 * unk_tgt / n_tgt:.2f}% <unk>)")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
