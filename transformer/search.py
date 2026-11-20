"""Decoding: greedy search and beam search (beam 5 for IWSLT'14, as in fairseq-generate).

Beam search keeps the `beam` best partial translations per sentence. At each step every
hypothesis is extended by every word, the 2*beam best candidates are kept, candidates that
end with </s> are moved to the finished list, and the best `beam` others continue.
A sentence is done when it has `beam` finished hypotheses. Finished hypotheses are ranked
by  sum(log p) / length**lenpen  (length normalisation, fairseq default lenpen = 1).
"""

from __future__ import annotations

import torch

from .model import Transformer


@torch.no_grad()
def greedy_search(model: Transformer, src_tokens: torch.Tensor, bos: int, eos: int, pad: int,
                  max_len: int) -> list[list[int]]:
    memory, src_pad_mask = model.encode(src_tokens)
    B = src_tokens.shape[0]
    cache = model.new_cache()
    tokens = torch.full((B, 1), bos, dtype=torch.long, device=src_tokens.device)
    done = torch.zeros(B, dtype=torch.bool, device=src_tokens.device)
    out = [[] for _ in range(B)]
    for step in range(max_len + 1):
        lprobs = torch.log_softmax(model.decode_step(tokens[:, -1:], step, memory, src_pad_mask,
                                                     cache).float(), dim=-1)
        lprobs[:, [pad, bos]] = float("-inf")
        if step == max_len:
            lprobs[:, :] = float("-inf")
            lprobs[:, eos] = 0.0
        nxt = lprobs.argmax(dim=-1)
        for b in range(B):
            if not done[b] and nxt[b] != eos:
                out[b].append(int(nxt[b]))
        done |= nxt.eq(eos)
        tokens = torch.cat([tokens, nxt[:, None]], dim=1)
        if done.all():
            break
    return out


@torch.no_grad()
def beam_search(model: Transformer, src_tokens: torch.Tensor, beam: int, bos: int, eos: int,
                pad: int, max_len_a: float = 0.0, max_len_b: int = 200,
                lenpen: float = 1.0) -> list[list[int]]:
    """Returns the best translation (token ids, without </s>) for each source sentence."""
    device = src_tokens.device
    B = src_tokens.shape[0]
    src_len = int(src_tokens.ne(pad).sum(dim=1).max())
    max_len = min(int(max_len_a * src_len + max_len_b), model.max_positions - 2)

    memory, src_pad_mask = model.encode(src_tokens)
    expand = torch.arange(B, device=device).repeat_interleave(beam)       # row b*beam+k <- b
    memory, src_pad_mask = memory[expand], src_pad_mask[expand]
    cache = model.new_cache()

    tokens = torch.full((B * beam, 1), bos, dtype=torch.long, device=device)
    beam_scores = torch.zeros(B, beam, device=device)
    beam_scores[:, 1:] = float("-inf")             # step 0: all beams are identical, keep one
    finished: list[list[tuple[float, list[int]]]] = [[] for _ in range(B)]
    done = [False] * B
    offsets = (torch.arange(B, device=device) * beam)[:, None]               # (B, 1)

    for step in range(max_len + 1):
        logits = model.decode_step(tokens[:, -1:], step, memory, src_pad_mask, cache)
        lprobs = torch.log_softmax(logits.float(), dim=-1)                   # (B*beam, V)
        lprobs[:, [pad, bos]] = float("-inf")
        if step == max_len:                       # out of room: every hypothesis must end now
            eos_col = lprobs[:, eos].clone()
            lprobs[:, :] = float("-inf")
            lprobs[:, eos] = eos_col
        V = lprobs.shape[1]

        cand = (beam_scores.view(-1, 1) + lprobs).view(B, beam * V)
        top_scores, top_idx = cand.topk(2 * beam, dim=1)                     # (B, 2*beam)
        top_beam, top_tok = top_idx // V, top_idx % V
        is_eos = top_tok.eq(eos) & torch.isfinite(top_scores)

        # 1) Finished hypotheses: </s> among the best `beam` candidates.
        for b, k in is_eos[:, :beam].nonzero().tolist():
            if done[b]:
                continue
            row = int(offsets[b, 0] + top_beam[b, k])
            hyp = tokens[row, 1:].tolist()                   # drop the leading <s>
            score = float(top_scores[b, k]) / (len(hyp) + 1) ** lenpen
            finished[b].append((score, hyp))
            if len(finished[b]) >= beam:
                done[b] = True
        if all(done):
            break

        # 2) Continuing hypotheses: the best `beam` candidates that are not </s>.
        #    Sort key: EOS candidates are pushed to the back, otherwise keep score order.
        position = torch.arange(2 * beam, device=device).expand(B, -1)
        sort_key = is_eos.long() * (2 * beam) + position
        keep = sort_key.topk(beam, dim=1, largest=False).indices             # (B, beam)
        new_rows = (offsets + top_beam.gather(1, keep)).view(-1)             # source hypothesis
        new_tok = top_tok.gather(1, keep).view(-1, 1)
        beam_scores = top_scores.gather(1, keep)
        beam_scores = beam_scores.masked_fill(torch.tensor(done, device=device)[:, None],
                                              float("-inf"))

        tokens = torch.cat([tokens[new_rows], new_tok], dim=1)
        model.reorder_cache(cache, new_rows)

    best = []
    for b in range(B):
        if finished[b]:
            best.append(max(finished[b], key=lambda s_h: s_h[0])[1])
        else:                                     # cannot happen in practice; be safe
            best.append([])
    return best
