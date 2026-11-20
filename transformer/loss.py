"""Label-smoothed cross-entropy (Sec. 5.4 of the paper, epsilon_ls = 0.1), as in fairseq.

Instead of a one-hot target, the model is trained towards
    (1 - eps) on the correct word + eps spread uniformly over the vocabulary.
This "hurts perplexity, as the model learns to be more unsure, but improves accuracy
and BLEU score" (paper, Sec. 5.4).
"""

from __future__ import annotations

import torch


def label_smoothed_nll_loss(lprobs: torch.Tensor, target: torch.Tensor, epsilon: float,
                            ignore_index: int):
    """lprobs: (N, V) log-probabilities, target: (N,). Returns (loss, nll_loss), both summed
    over non-padding positions. nll_loss is the plain cross-entropy, reported for perplexity."""
    target = target.unsqueeze(-1)
    nll = -lprobs.gather(dim=-1, index=target)            # -log p(correct word)
    smooth = -lprobs.sum(dim=-1, keepdim=True)            # -sum_v log p(v)
    pad = target.eq(ignore_index)
    nll, smooth = nll.masked_fill(pad, 0.0), smooth.masked_fill(pad, 0.0)
    nll, smooth = nll.sum(), smooth.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll + eps_i * smooth
    return loss, nll
