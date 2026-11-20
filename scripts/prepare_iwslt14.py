"""Python version of fairseq's prepare-iwslt14.sh, for machines without Perl or git.

Same steps, same order, same output files:
  1. download de-en.tgz (skipped if it is already in --orig) and unpack it
  2. train: drop <url>/<talkid>/<keywords> lines, strip <title>/<description> tags,
     Moses-tokenise, clean (1-175 tokens, length ratio <= 1.5), lower-case
  3. dev/test XML: keep <seg> lines, replace ’ by ', tokenise, lower-case
  4. valid = every 23rd training line, train = the rest,
     test = dev2010 + TEDX dev2012 + tst2010 + tst2011 + tst2012
  5. learn joint BPE with 10,000 merges on train.de + train.en and apply it to every file

Moses tokenisation uses `sacremoses`, a Python port of tokenizer.perl. The bash script
remains the reference; tests/test_prepare.py checks both give identical files on toy data.

    python scripts/prepare_iwslt14.py --orig data/orig --out data/iwslt14.tokenized.de-en
"""

from __future__ import annotations

import argparse
import re
import shutil
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath

from sacremoses import MosesTokenizer
from subword_nmt.apply_bpe import BPE
from subword_nmt.learn_bpe import learn_bpe

URL = "https://dl.fbaipublicfiles.com/fairseq/data/iwslt14/de-en.tgz"
SRC, TGT, LANG = "de", "en", "de-en"
TEST_SETS = ["IWSLT14.TED.dev2010", "IWSLT14.TEDX.dev2012", "IWSLT14.TED.tst2010",
             "IWSLT14.TED.tst2011", "IWSLT14.TED.tst2012"]


def read_lines(path: Path) -> list[str]:
    # Only "\n" ends a line, exactly as for grep/sed/Perl in the bash script. (Python's
    # default would also break lines at a stray "\r"; the tokenizer turns it into a space.)
    with open(path, encoding="utf-8", newline="\n") as f:
        return [line.rstrip("\n") for line in f]


def write_lines(path: Path, lines: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(line + "\n" for line in lines)


def tokenize(lines: list[str], lang: str) -> list[str]:
    tok = MosesTokenizer(lang=lang)
    return [tok.tokenize(line, return_str=True, escape=True) for line in lines]


def train_lines(path: Path) -> list[str]:
    out = []
    for line in read_lines(path):
        if "<url>" in line or "<talkid>" in line or "<keywords>" in line:
            continue
        for tag in ("<title>", "</title>", "<description>", "</description>"):
            line = line.replace(tag, "")
        out.append(line)
    return out


def clean_corpus(src: list[str], tgt: list[str], min_len=1, max_len=175, ratio=1.5,
                 max_word_length=1000):
    """Port of Moses clean-corpus-n.perl (the options used by the fairseq script)."""
    if not src or len(src) != len(tgt):
        raise ValueError("The training corpus must contain matching nonempty source and target files.")
    factored = "|" in src[0] or "|" in tgt[0]
    keep_s, keep_t = [], []
    for f, e in zip(src, tgt):
        if not factored:
            f, e = f.replace("|", ""), e.replace("|", "")
        f = re.sub(r"\s+", " ", f)
        e = re.sub(r"\s+", " ", e)
        f = f[1:] if f.startswith(" ") else f
        e = e[1:] if e.startswith(" ") else e
        f = f[:-1] if f.endswith(" ") else f
        e = e[:-1] if e.endswith(" ") else e
        if not f or not e:
            continue
        fc, ec = len(f.split(" ")), len(e.split(" "))
        if ec > max_len or fc > max_len or ec < min_len or fc < min_len:
            continue
        if ec / fc > ratio or fc / ec > ratio:
            continue
        too_long = re.compile(r"[^\s|]{%d}" % (max_word_length + 1))
        if too_long.search(e) or too_long.search(f):
            continue
        keep_s.append(f)
        keep_t.append(e)
    return keep_s, keep_t


def xml_lines(path: Path) -> list[str]:
    out = []
    for line in read_lines(path):
        if "<seg id" not in line:
            continue
        line = re.sub(r'<seg id="[0-9]*">\s*', "", line)
        line = re.sub(r"\s*</seg>\s*", "", line)
        out.append(line.replace("’", "'"))
    return out


def extract_corpus(tar: tarfile.TarFile, dest: Path) -> None:
    if hasattr(tarfile, "data_filter"):
        tar.extractall(dest, filter="data")
        return
    root = dest.resolve()
    for member in tar:
        name = PurePosixPath(member.name)
        if name.is_absolute() or ".." in name.parts or not (member.isdir() or member.isfile()):
            raise ValueError(f"Unsafe corpus archive entry: {member.name!r}")
        target = root.joinpath(*name.parts)
        if not target.resolve().is_relative_to(root):
            raise ValueError(f"Corpus archive entry escapes the output directory: {member.name!r}")
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(member)
            if src is None:
                raise ValueError(f"Unreadable corpus archive entry: {member.name!r}")
            temp = target.with_name(target.name + ".part")
            with src, temp.open("xb") as output:
                shutil.copyfileobj(src, output)
            temp.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--orig", default="data/orig")
    parser.add_argument("--out", default="data/iwslt14.tokenized.de-en")
    parser.add_argument("--bpe-tokens", type=int, default=10000)
    args = parser.parse_args()

    orig, prep = Path(args.orig), Path(args.out)
    tmp = prep / "tmp"
    for d in (orig, tmp):
        d.mkdir(parents=True, exist_ok=True)

    gz = orig / "de-en.tgz"
    if not gz.exists():
        print(f"downloading {URL} ...")
        part = gz.with_suffix(".tgz.part")
        try:
            urllib.request.urlretrieve(URL, part)
            part.replace(gz)
        finally:
            part.unlink(missing_ok=True)
    with tarfile.open(gz) as tar:
        extract_corpus(tar, orig)
    raw = orig / LANG

    print("pre-processing train data ...")
    tok = {l: tokenize(train_lines(raw / f"train.tags.{LANG}.{l}"), l) for l in (SRC, TGT)}
    clean_src, clean_tgt = clean_corpus(tok[SRC], tok[TGT])
    train_all = {SRC: [s.lower() for s in clean_src], TGT: [t.lower() for t in clean_tgt]}

    print("pre-processing valid/test data ...")
    test = {SRC: [], TGT: []}
    for name in TEST_SETS:
        subset = {}
        for l in (SRC, TGT):
            lines = [s.lower() for s in tokenize(xml_lines(raw / f"{name}.{LANG}.{l}.xml"), l)]
            write_lines(tmp / f"{name}.{LANG}.{l}", lines)
            subset[l] = lines
        if not subset[SRC] or len(subset[SRC]) != len(subset[TGT]):
            raise ValueError(f"Unaligned or empty source and target files in {name}.")
        for l, lines in subset.items():
            test[l] += lines

    print("creating train, valid, test ...")
    splits = {}
    for l in (SRC, TGT):
        lines = train_all[l]
        splits[("valid", l)] = [x for i, x in enumerate(lines, 1) if i % 23 == 0]
        splits[("train", l)] = [x for i, x in enumerate(lines, 1) if i % 23 != 0]
        splits[("test", l)] = test[l]
        for split in ("train", "valid", "test"):
            write_lines(tmp / f"{split}.{l}", splits[(split, l)])

    print(f"learning BPE ({args.bpe_tokens} merges) on train.{SRC} + train.{TGT} ...")
    joint = tmp / f"train.{TGT}-{SRC}"
    write_lines(joint, splits[("train", SRC)] + splits[("train", TGT)])
    code = prep / "code"
    with open(joint, encoding="utf-8") as fin, open(code, "w", encoding="utf-8") as fout:
        learn_bpe(fin, fout, args.bpe_tokens)
    with open(code, encoding="utf-8") as f:
        bpe = BPE(f)
    for (split, l), lines in splits.items():
        write_lines(prep / f"{split}.{l}", [bpe.process_line(x + "\n").rstrip("\n")
                                             for x in lines])
    for split in ("train", "valid", "test"):
        print(f"{split}: {len(splits[(split, SRC)])} sentence pairs")
    print(f"done: {prep}")


if __name__ == "__main__":
    main()
