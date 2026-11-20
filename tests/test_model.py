import math

import torch
import torch.nn.functional as F

from transformer.model import MultiHeadAttention, Transformer, causal_mask, sinusoidal_positions

PAD = 1


def tiny(seed=0, **kw):
    torch.manual_seed(seed)
    args = dict(src_vocab=40, tgt_vocab=48, pad_idx=PAD, d_model=32, d_ff=64, n_heads=4,
                enc_layers=2, dec_layers=2, dropout=0.0)
    args.update(kw)
    return Transformer(**args).eval()


def test_positional_encoding_formula():
    pe = sinusoidal_positions(50, 16)
    pos, i = 7, 3
    assert math.isclose(pe[pos, 2 * i].item(), math.sin(pos / 10000 ** (2 * i / 16)), abs_tol=1e-6)
    angle = pos / 10000 ** (2 * i / 16)
    assert math.isclose(pe[pos, 2 * i + 1].item(), math.cos(angle), abs_tol=1e-6)


def test_attention_matches_pytorch_reference():
    torch.manual_seed(0)
    mha = MultiHeadAttention(32, 4).eval()
    q, kv = torch.randn(2, 5, 32), torch.randn(2, 7, 32)
    pad = torch.zeros(2, 7, dtype=torch.bool)
    pad[1, 5:] = True
    ours = mha(q, kv, kv, key_padding_mask=pad)

    def heads(x):
        return x.view(2, -1, 4, 8).transpose(1, 2)
    Q, K, V = heads(mha.q_proj(q)), heads(mha.k_proj(kv)), heads(mha.v_proj(kv))
    ref = F.scaled_dot_product_attention(Q, K, V, attn_mask=~pad[:, None, None, :])
    ref = mha.out_proj(ref.transpose(1, 2).reshape(2, 5, 32))
    assert torch.allclose(ours, ref, atol=1e-5)


def test_causal_mask_blocks_the_future():
    m = tiny()
    src = torch.randint(4, 40, (2, 6))
    prev = torch.randint(4, 48, (2, 5))
    out1 = m(src, prev)
    prev2 = prev.clone()
    prev2[:, 3:] = torch.randint(4, 48, (2, 2))       # change only the last two positions
    out2 = m(src, prev2)
    assert torch.allclose(out1[:, :3], out2[:, :3], atol=1e-5)   # earlier positions unaffected
    assert not torch.allclose(out1[:, 3:], out2[:, 3:])
    assert causal_mask(3).tolist() == [[False, True, True], [False, False, True],
                                       [False, False, False]]


def test_source_padding_is_ignored():
    m = tiny()
    src = torch.randint(4, 40, (1, 6))
    padded = torch.cat([src, torch.full((1, 3), PAD)], dim=1)
    prev = torch.randint(4, 48, (1, 4))
    assert torch.allclose(m(src, prev), m(padded, prev), atol=1e-5)


def test_incremental_decoding_matches_full_decoding():
    m = tiny()
    src = torch.randint(4, 40, (3, 7))
    prev = torch.randint(4, 48, (3, 6))
    full = m(src, prev)
    memory, src_pad = m.encode(src)
    cache = m.new_cache()
    steps = [m.decode_step(prev[:, t:t + 1], t, memory, src_pad, cache) for t in range(6)]
    assert torch.allclose(torch.stack(steps, dim=1), full, atol=1e-5)


def test_output_layer_is_tied_to_the_target_embedding():
    m = tiny()
    names = [n for n, _ in m.named_parameters()]
    assert "tgt_embed.weight" in names and not any("out" in n and "proj" not in n for n in names)


def test_iwslt_model_size():
    m = Transformer(8848, 6632, PAD)          # vocab sizes of the real IWSLT'14 dictionaries
    n = sum(p.numel() for p in m.parameters())
    emb = (8848 + 6632) * 512
    enc = 6 * (4 * (512 * 512 + 512) + (512 * 1024 + 1024) + (1024 * 512 + 512) + 2 * 2 * 512)
    dec = 6 * (8 * (512 * 512 + 512) + (512 * 1024 + 1024) + (1024 * 512 + 512) + 3 * 2 * 512)
    assert n == emb + enc + dec
    assert 39e6 < n < 40e6
