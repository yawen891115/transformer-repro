"""Vocabulary, binarised parallel data and token-based batching.

Text files are already tokenised, lower-cased and BPE-segmented by
scripts/prepare-iwslt14.sh, one sentence per line, e.g.

    train.de:  wir müssen ver@@ stehen , dass ...
    train.en:  we have to understand that ...

`preprocess.py` turns them into integer tensors with these classes; `train.py` and
`generate.py` read the tensors back.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import torch

BOS, PAD, EOS, UNK = "<s>", "<pad>", "</s>", "<unk>"


def read_lines(path: Path) -> list[str]:
    """Lines of a UTF-8 text file. Only "\n" ends a line (newline="\n"), as in the Perl/sed
    tools that wrote the files; Python's default would also split at a stray "\r" and
    shift the German and English sides against each other."""
    with open(path, encoding="utf-8", newline="\n") as f:
        return [line.rstrip("\n") for line in f]


class Dictionary:
    """Word <-> id mapping. Ids 0-3 are the special symbols (same order as fairseq)."""

    def __init__(self):
        self.symbols = [BOS, PAD, EOS, UNK]
        self.counts = [0, 0, 0, 0]
        self.index = {s: i for i, s in enumerate(self.symbols)}

    bos, pad, eos, unk = 0, 1, 2, 3

    def __len__(self):
        return len(self.symbols)

    def add(self, word: str, count: int = 1) -> None:
        if word in self.index:
            self.counts[self.index[word]] += count
        else:
            self.index[word] = len(self.symbols)
            self.symbols.append(word)
            self.counts.append(count)

    @classmethod
    def build(cls, files: list[Path], padding_factor: int = 8) -> "Dictionary":
        """Every word of the training files, most frequent first (ties: alphabetical).
        The size is padded to a multiple of 8, which helps GPU matrix multiplies."""
        counter = Counter()
        for path in files:
            for line in read_lines(path):
                counter.update(line.split())
        d = cls()
        for word, count in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
            d.add(word, count)
        i = 0
        while len(d) % padding_factor != 0:
            d.add(f"madeupword{i:04d}", 0)
            i += 1
        return d

    def encode(self, line: str) -> torch.Tensor:
        """'a b c' -> tensor([id(a), id(b), id(c), eos])."""
        ids = [self.index.get(w, self.unk) for w in line.split()] + [self.eos]
        return torch.tensor(ids, dtype=torch.int32)

    def decode(self, ids, remove_bpe: bool = False, escape_unk: bool = False) -> str:
        """ids -> text. For references use escape_unk=True so an <unk> in the reference can
        never be matched by an <unk> in the hypothesis (fairseq does the same)."""
        words = [("<<unk>>" if escape_unk and i == self.unk else self.symbols[i])
                 for i in ids if i not in (self.bos, self.pad, self.eos)]
        text = " ".join(words)
        return remove_bpe_marks(text) if remove_bpe else text

    def save(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for sym, cnt in zip(self.symbols[4:], self.counts[4:]):
                f.write(f"{sym} {cnt}\n")

    @classmethod
    def load(cls, path: Path) -> "Dictionary":
        d = cls()
        with open(path, encoding="utf-8") as f:
            for line in f:
                word, count = line.rstrip("\n").rsplit(" ", 1)
                d.add(word, int(count))
        return d


def remove_bpe_marks(text: str) -> str:
    """Undo subword-nmt segmentation: 'ver@@ stehen' -> 'verstehen' (fairseq --remove-bpe)."""
    return (text + " ").replace("@@ ", "").rstrip()


class ParallelData:
    """Sentence pairs as two lists of id tensors (each ending with </s>) plus their lengths."""

    def __init__(self, src: list[torch.Tensor], tgt: list[torch.Tensor]):
        assert len(src) == len(tgt)
        self.src, self.tgt = src, tgt
        self.src_lens = np.array([len(s) for s in src])
        self.tgt_lens = np.array([len(t) for t in tgt])

    def __len__(self):
        return len(self.src)

    @classmethod
    def from_text(cls, src_file: Path, tgt_file: Path, src_dict: Dictionary,
                  tgt_dict: Dictionary) -> "ParallelData":
        src_lines, tgt_lines = read_lines(src_file), read_lines(tgt_file)
        if len(src_lines) != len(tgt_lines):
            raise ValueError(f"{src_file} has {len(src_lines)} lines but {tgt_file} has "
                             f"{len(tgt_lines)}; the two sides must be aligned line by line")
        return cls([src_dict.encode(a) for a in src_lines], [tgt_dict.encode(b) for b in tgt_lines])

    def save(self, path: Path) -> None:
        torch.save({"src": self.src, "tgt": self.tgt}, path)

    @classmethod
    def load(cls, path: Path) -> "ParallelData":
        data = torch.load(path)
        return cls(data["src"], data["tgt"])


def make_batches(data: ParallelData, max_tokens: int, shuffle: bool, seed: int = 0,
                 max_sentences: int | None = None) -> list[np.ndarray]:
    """Group sentence pairs of similar length so that every batch holds at most `max_tokens`
    tokens (counting padding): batch_size * max(src_len, tgt_len) <= max_tokens.
    Like fairseq: random order first, then a stable sort by length, then shuffle the batches."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(data)) if shuffle else np.arange(len(data))
    order = order[np.argsort(data.tgt_lens[order], kind="mergesort")]
    order = order[np.argsort(data.src_lens[order], kind="mergesort")]

    batches, current, longest = [], [], 0
    for idx in order:
        n = max(data.src_lens[idx], data.tgt_lens[idx])
        new_longest = max(longest, n)
        too_many = max_sentences is not None and len(current) >= max_sentences
        if current and (new_longest * (len(current) + 1) > max_tokens or too_many):
            batches.append(np.array(current))
            current, new_longest = [], n
        current.append(idx)
        longest = new_longest
    if current:
        batches.append(np.array(current))
    if shuffle:
        rng.shuffle(batches)
    return batches


def collate(data: ParallelData, indices: np.ndarray, pad: int, bos: int) -> dict:
    """Pad a batch. The decoder input is the target shifted right with <s> in front:
        target          = [y1, y2, y3, </s>]
        prev_tokens     = [<s>, y1, y2, y3]      (teacher forcing)"""
    src = [data.src[i].long() for i in indices]
    tgt = [data.tgt[i].long() for i in indices]
    B = len(indices)
    src_tokens = torch.full((B, max(len(s) for s in src)), pad, dtype=torch.long)
    target = torch.full((B, max(len(t) for t in tgt)), pad, dtype=torch.long)
    prev = torch.full_like(target, pad)
    for k, (s, t) in enumerate(zip(src, tgt)):
        src_tokens[k, : len(s)] = s
        target[k, : len(t)] = t
        prev[k, 0] = bos
        prev[k, 1: len(t)] = t[:-1]
    return {"src_tokens": src_tokens, "prev_tokens": prev, "target": target,
            "ntokens": int(target.ne(pad).sum()), "indices": indices}
