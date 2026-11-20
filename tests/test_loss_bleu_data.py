import math
import random

import numpy as np
import sacrebleu
import torch

from transformer.bleu import corpus_bleu
from transformer.data import Dictionary, ParallelData, collate, make_batches, remove_bpe_marks
from transformer.loss import label_smoothed_nll_loss


def test_label_smoothing_target_distribution():
    torch.manual_seed(0)
    V, eps = 10, 0.1
    lprobs = torch.log_softmax(torch.randn(4, V), -1)
    target = torch.tensor([2, 5, 1, 7])                    # 1 = padding
    loss, nll = label_smoothed_nll_loss(lprobs, target, eps, ignore_index=1)
    keep = target != 1
    q = torch.full((4, V), eps / (V - 1))                  # eps spread over the other words
    q[torch.arange(4), target] = 1 - eps                   # 1 - eps on the correct word
    expected = -(q * lprobs).sum(-1)[keep].sum()
    assert torch.allclose(loss, expected, atol=1e-5)
    assert torch.allclose(nll, -lprobs[torch.arange(4), target][keep].sum())
    loss0, nll0 = label_smoothed_nll_loss(lprobs, target, 0.0, ignore_index=1)
    assert torch.allclose(loss0, nll0)                     # eps = 0 -> plain cross-entropy


def test_bleu_matches_sacrebleu():
    rng = random.Random(0)
    words = [f"w{i}" for i in range(30)]
    refs = [" ".join(rng.choices(words, k=rng.randint(5, 20))) for _ in range(200)]
    hyps = []
    for r in refs:
        toks = r.split()
        toks = [t if rng.random() < 0.7 else rng.choice(words) for t in toks]
        hyps.append(" ".join(toks[: max(1, len(toks) - rng.randint(0, 3))]))
    ours = corpus_bleu(hyps, refs)["bleu"]
    ref = sacrebleu.corpus_bleu(hyps, [refs], tokenize="none", force=True).score
    assert abs(ours - ref) < 1e-6
    assert corpus_bleu(refs, refs)["bleu"] == 100.0


def test_bleu_brevity_penalty():
    s = corpus_bleu(["a b c d"], ["a b c d e f g h"])
    assert math.isclose(s["bp"], math.exp(1 - 8 / 4))


def test_dictionary_and_bpe_removal(tmp_path):
    f = tmp_path / "train.txt"
    f.write_text("b a\nb c\nb\n", encoding="utf-8")
    d = Dictionary.build([f])
    assert d.symbols[:7] == ["<s>", "<pad>", "</s>", "<unk>", "b", "a", "c"]
    assert len(d) % 8 == 0
    ids = d.encode("a zzz c")
    assert ids.tolist() == [5, d.unk, 6, d.eos]
    assert d.decode(ids.tolist()) == "a <unk> c"
    assert d.decode(ids.tolist(), escape_unk=True) == "a <<unk>> c"
    assert remove_bpe_marks("wir ver@@ stehen das ge@@ setz") == "wir verstehen das gesetz"
    d.save(tmp_path / "dict.txt")
    assert Dictionary.load(tmp_path / "dict.txt").symbols == d.symbols


def test_batches_respect_max_tokens_and_cover_everything():
    rng = np.random.default_rng(0)
    src = [torch.ones(int(n), dtype=torch.int32) for n in rng.integers(2, 60, 500)]
    tgt = [torch.ones(int(n), dtype=torch.int32) for n in rng.integers(2, 60, 500)]
    data = ParallelData(src, tgt)
    batches = make_batches(data, max_tokens=256, shuffle=True, seed=3)
    seen = np.concatenate(batches)
    assert sorted(seen.tolist()) == list(range(500))
    for b in batches:
        longest = max(max(data.src_lens[i], data.tgt_lens[i]) for i in b)
        assert longest * len(b) <= 256 or len(b) == 1


def test_collate_shifts_the_target_right():
    data = ParallelData([torch.tensor([10, 11, 2])], [torch.tensor([20, 21, 22, 2])])
    b = collate(data, np.array([0]), pad=1, bos=0)
    assert b["target"].tolist() == [[20, 21, 22, 2]]
    assert b["prev_tokens"].tolist() == [[0, 20, 21, 22]]
    assert b["ntokens"] == 4
