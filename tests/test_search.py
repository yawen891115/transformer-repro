import torch

from transformer.model import Transformer
from transformer.search import beam_search, greedy_search

BOS, PAD, EOS = 0, 1, 2


def model():
    torch.manual_seed(0)
    return Transformer(30, 30, PAD, d_model=32, d_ff=64, n_heads=4, enc_layers=1,
                       dec_layers=2, dropout=0.0).eval()


def test_beam_size_one_equals_greedy():
    m = model()
    src = torch.randint(4, 30, (4, 6))
    g = greedy_search(m, src, BOS, EOS, PAD, max_len=12)
    b = beam_search(m, src, 1, BOS, EOS, PAD, max_len_a=0, max_len_b=12)
    assert g == b


def test_beam_outputs_are_clean_and_bounded():
    m = model()
    src = torch.randint(4, 30, (5, 8))
    for out in beam_search(m, src, 5, BOS, EOS, PAD, max_len_a=0, max_len_b=10):
        assert len(out) <= 10
        assert all(t not in (BOS, PAD, EOS) for t in out)


def score(m, src, hyp):
    """Length-normalised log-probability of a full hypothesis (+ </s>) under the model."""
    prev = torch.tensor([[BOS] + hyp])
    tgt = torch.tensor([hyp + [EOS]])
    lp = torch.log_softmax(m(src, prev), -1).gather(-1, tgt[..., None]).sum()
    return lp.item() / (len(hyp) + 1)


def test_beam_finds_at_least_as_good_a_translation_as_greedy():
    m = model()
    wins = 0
    for seed in range(10):
        torch.manual_seed(seed)
        src = torch.randint(4, 30, (1, 6))
        g = greedy_search(m, src, BOS, EOS, PAD, max_len=15)[0]
        b = beam_search(m, src, 5, BOS, EOS, PAD, max_len_a=0, max_len_b=15)[0]
        wins += score(m, src, b) >= score(m, src, g) - 1e-5
    assert wins >= 9
