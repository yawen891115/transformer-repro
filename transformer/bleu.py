"""Corpus-level BLEU (Papineni et al., 2002), the same computation as Moses' multi-bleu.perl.

    BLEU = BP * exp( (1/4) * sum_{n=1..4} log p_n )
    p_n  = clipped n-gram matches / n-grams in the hypotheses     (summed over the corpus)
    BP   = 1 if hyp_len > ref_len else exp(1 - ref_len / hyp_len)  (brevity penalty)

Inputs are whitespace-tokenised strings. For IWSLT'14 we score tokenised, lower-cased text
after removing BPE, which is what `fairseq-generate --remove-bpe` reports.
"""

from __future__ import annotations

import math
from collections import Counter


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(hypotheses: list[str], references: list[str], max_n: int = 4) -> dict:
    assert len(hypotheses) == len(references)
    matches, totals = [0] * max_n, [0] * max_n
    hyp_len = ref_len = 0
    for hyp, ref in zip(hypotheses, references):
        h, r = hyp.split(), ref.split()
        hyp_len, ref_len = hyp_len + len(h), ref_len + len(r)
        for n in range(1, max_n + 1):
            h_ngrams, r_ngrams = _ngrams(h, n), _ngrams(r, n)
            matches[n - 1] += sum(min(c, r_ngrams[g]) for g, c in h_ngrams.items())
            totals[n - 1] += max(len(h) - n + 1, 0)

    precisions = [100.0 * m / t if t > 0 else 0.0 for m, t in zip(matches, totals)]
    if min(matches) == 0 or hyp_len == 0:
        score = 0.0
    else:
        log_p = sum(math.log(m / t) for m, t in zip(matches, totals)) / max_n
        bp = 1.0 if hyp_len > ref_len else math.exp(1.0 - ref_len / hyp_len)
        score = 100.0 * bp * math.exp(log_p)
    bp = 1.0 if hyp_len > ref_len or hyp_len == 0 else math.exp(1.0 - ref_len / hyp_len)
    return {"bleu": score, "precisions": precisions, "bp": bp,
            "hyp_len": hyp_len, "ref_len": ref_len}
