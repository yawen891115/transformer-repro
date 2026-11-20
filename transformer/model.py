"""The Transformer of "Attention Is All You Need" (Vaswani et al., 2017), written from scratch.

    source tokens -> embedding * sqrt(d) + positional encoding -> N x EncoderLayer
    target tokens (shifted right) -> embedding * sqrt(d) + positional encoding
                  -> N x DecoderLayer (causal self-attention, cross-attention to the encoder)
                  -> linear layer tied to the target embedding -> softmax over the vocabulary

Every sub-layer is wrapped as LayerNorm(x + Dropout(Sublayer(x))) ("post-norm", Sec. 5.4).
Sizes for IWSLT'14 De->En follow fairseq's `transformer_iwslt_de_en`: d_model 512,
FFN 1024, 4 heads, 6 + 6 layers, dropout 0.3.

The decoder also supports incremental decoding with a key/value cache (used by beam
search): each new token only attends to cached keys/values instead of recomputing the prefix.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- attention
class MultiHeadAttention(nn.Module):
    """MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O,
    head_i = softmax(Q_i K_i^T / sqrt(d_k)) V_i   (Sec. 3.2)."""

    def __init__(self, d_model: int, n_heads: int, attn_dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(attn_dropout)
        # Initialisation used by fairseq.
        for proj in (self.q_proj, self.k_proj, self.v_proj):
            nn.init.xavier_uniform_(proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _split(self, x: torch.Tensor) -> torch.Tensor:          # (B, T, D) -> (B, H, T, d_head)
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.d_head).transpose(1, 2)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None,
                cache: dict | None = None, static_kv: bool = False) -> torch.Tensor:
        """
        query (B, Tq, D); key, value (B, Tk, D)
        key_padding_mask (B, Tk) bool, True = padding (ignored)
        attn_mask (Tq, Tk) bool, True = not allowed (the causal mask)
        cache: dict for incremental decoding. Self-attention appends the new key/value;
               cross-attention (static_kv=True) computes the encoder key/value once.
        """
        q = self._split(self.q_proj(query))
        if cache is not None and static_kv and "k" in cache:
            k, v = cache["k"], cache["v"]
        else:
            k, v = self._split(self.k_proj(key)), self._split(self.v_proj(value))
            if cache is not None:
                if not static_kv and "k" in cache:
                    k = torch.cat([cache["k"], k], dim=2)
                    v = torch.cat([cache["v"], v], dim=2)
                cache["k"], cache["v"] = k, v

        scores = q @ k.transpose(-1, -2) / math.sqrt(self.d_head)        # (B, H, Tq, Tk)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        attn = torch.softmax(scores.float(), dim=-1).type_as(scores)
        out = self.attn_dropout(attn) @ v                                 # (B, H, Tq, d_head)
        out = out.transpose(1, 2).reshape(query.shape[0], query.shape[1], -1)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """FFN(x) = max(0, x W1 + b1) W2 + b2  (Eq. 2)."""

    def __init__(self, d_model: int, d_ff: int, act_dropout: float = 0.0):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(d_model, d_ff), nn.Linear(d_ff, d_model)
        self.act_dropout = nn.Dropout(act_dropout)

    def forward(self, x):
        return self.fc2(self.act_dropout(F.relu(self.fc1(x))))


def sinusoidal_positions(max_len: int, d_model: int) -> torch.Tensor:
    """Sec. 3.5: PE(pos, 2i) = sin(pos / 10000^(2i/d)), PE(pos, 2i+1) = cos(pos / 10000^(2i/d))."""
    pos = torch.arange(max_len, dtype=torch.float32)[:, None]
    two_i = torch.arange(0, d_model, 2, dtype=torch.float32)
    div = torch.exp(two_i * (-math.log(10000.0) / d_model))           # 1 / 10000^(2i/d)
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def causal_mask(size: int, device=None) -> torch.Tensor:
    """True above the diagonal: position i may not look at positions j > i."""
    return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)


# --------------------------------------------------------------------------- layers
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, attn_dropout=0.0, act_dropout=0.0):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, attn_dropout)
        self.ffn = FeedForward(d_model, d_ff, act_dropout)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_pad_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, key_padding_mask=src_pad_mask)))
        return self.norm2(x + self.dropout(self.ffn(x)))


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, attn_dropout=0.0, act_dropout=0.0):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, attn_dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, attn_dropout)
        self.ffn = FeedForward(d_model, d_ff, act_dropout)
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(d_model) for _ in range(3))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_pad_mask, self_attn_mask=None, tgt_pad_mask=None,
                cache: dict | None = None):
        sa = self.self_attn(x, x, x, key_padding_mask=tgt_pad_mask, attn_mask=self_attn_mask,
                            cache=None if cache is None else cache["self"])
        x = self.norm1(x + self.dropout(sa))
        ca = self.cross_attn(x, memory, memory, key_padding_mask=src_pad_mask,
                             cache=None if cache is None else cache["cross"], static_kv=True)
        x = self.norm2(x + self.dropout(ca))
        return self.norm3(x + self.dropout(self.ffn(x)))


# --------------------------------------------------------------------------- model
class Transformer(nn.Module):
    def __init__(self, src_vocab: int, tgt_vocab: int, pad_idx: int, d_model: int = 512,
                 d_ff: int = 1024, n_heads: int = 4, enc_layers: int = 6, dec_layers: int = 6,
                 dropout: float = 0.3, attn_dropout: float = 0.0, act_dropout: float = 0.0,
                 max_positions: int = 1024, share_all_embeddings: bool = False):
        super().__init__()
        self.pad_idx, self.d_model, self.max_positions = pad_idx, d_model, max_positions
        self.embed_scale = math.sqrt(d_model)

        self.src_embed = nn.Embedding(src_vocab, d_model, padding_idx=pad_idx)
        if share_all_embeddings:
            assert src_vocab == tgt_vocab, "sharing all embeddings needs a joined dictionary"
            self.tgt_embed = self.src_embed
        else:
            self.tgt_embed = nn.Embedding(tgt_vocab, d_model, padding_idx=pad_idx)
        for emb in (self.src_embed, self.tgt_embed):        # fairseq's embedding initialisation
            nn.init.normal_(emb.weight, mean=0.0, std=d_model ** -0.5)
            with torch.no_grad():
                emb.weight[pad_idx].zero_()
        self.register_buffer("pos", sinusoidal_positions(max_positions, d_model), persistent=False)
        self.dropout = nn.Dropout(dropout)

        args = (d_model, n_heads, d_ff, dropout, attn_dropout, act_dropout)
        self.encoder = nn.ModuleList(EncoderLayer(*args) for _ in range(enc_layers))
        self.decoder = nn.ModuleList(DecoderLayer(*args) for _ in range(dec_layers))
        # Output projection shares its weight matrix with the target embedding (Sec. 3.4).

    def _embed(self, emb: nn.Embedding, tokens: torch.Tensor, start: int = 0) -> torch.Tensor:
        x = emb(tokens) * self.embed_scale + self.pos[start:start + tokens.shape[1]]
        return self.dropout(x)

    def _output(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.tgt_embed.weight.T                          # logits over the target vocab

    # ------------------------------------------------------------------ training path
    def encode(self, src_tokens: torch.Tensor):
        src_pad_mask = src_tokens.eq(self.pad_idx)
        x = self._embed(self.src_embed, src_tokens)
        for layer in self.encoder:
            x = layer(x, src_pad_mask)
        return x, src_pad_mask

    def decode(self, prev_tokens, memory, src_pad_mask) -> torch.Tensor:
        """Teacher forcing: all target positions at once, with the causal mask."""
        tgt_pad_mask = prev_tokens.eq(self.pad_idx)
        mask = causal_mask(prev_tokens.shape[1], prev_tokens.device)
        x = self._embed(self.tgt_embed, prev_tokens)
        for layer in self.decoder:
            x = layer(x, memory, src_pad_mask, mask, tgt_pad_mask)
        return self._output(x)

    def forward(self, src_tokens, prev_tokens) -> torch.Tensor:
        memory, src_pad_mask = self.encode(src_tokens)
        return self.decode(prev_tokens, memory, src_pad_mask)       # (B, T, V)

    # ------------------------------------------------------------------ inference path
    def new_cache(self) -> list:
        return [{"self": {}, "cross": {}} for _ in self.decoder]

    @staticmethod
    def reorder_cache(cache: list, order: torch.Tensor) -> None:
        """Beam search keeps/duplicates hypotheses: reorder every cached tensor the same way."""
        for layer in cache:
            for part in layer.values():
                for key in part:
                    part[key] = part[key].index_select(0, order)

    def decode_step(self, last_tokens, step: int, memory, src_pad_mask, cache: list):
        """Logits for the next token given only the newest token (B, 1) and the cache."""
        x = self._embed(self.tgt_embed, last_tokens, start=step)
        for layer, layer_cache in zip(self.decoder, cache):
            x = layer(x, memory, src_pad_mask, cache=layer_cache)  # no mask: only past keys exist
        return self._output(x)[:, -1]                               # (B, V)
