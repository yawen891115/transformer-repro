# Transformer for German → English translation

A PyTorch implementation of Vaswani et al.'s encoder–decoder Transformer, with a fairseq-style IWSLT'14 German → English recipe. Attention, masking, the decoder cache, and beam search are implemented directly instead of using `torch.nn.Transformer` or fairseq model code.

## Implementation

| Component | What to inspect |
| --- | --- |
| Multi-head attention, positional encoding, post-norm layers | [transformer/model.py](transformer/model.py) |
| Padding and causal masks, tied target embeddings, incremental KV cache | [transformer/model.py](transformer/model.py) |
| Label smoothing and inverse-square-root learning-rate schedule | [transformer/loss.py](transformer/loss.py), [transformer/utils.py](transformer/utils.py) |
| Batched beam search with length normalization and greedy decoding | [transformer/search.py](transformer/search.py) |
| Moses tokenization, joint 10,000-merge BPE, corpus binarization | [scripts/prepare_iwslt14.py](scripts/prepare_iwslt14.py), [preprocess.py](preprocess.py) |
| Validation-based checkpoint selection and BLEU reporting | [train.py](train.py), [generate.py](generate.py) |

The IWSLT recipe sets model width 512, feed-forward width 1,024, four attention heads, six layers per stack, 4,096 tokens per batch, and a beam of five. It is an IWSLT adaptation of the original paper's architecture, not the paper's WMT14 benchmark.

## Quick start

Use Python 3.12+ and install the dependencies. Run the commands from the repository root.

```bash
python -m pip install -r requirements.txt

# Small synthetic corpus: data preparation → training → decoding.
python scripts/make_toy_corpus.py --out data/toy/orig/de-en.tgz
python scripts/prepare_iwslt14.py --orig data/toy/orig --out data/toy/iwslt14.tokenized.de-en
python preprocess.py --textdir data/toy/iwslt14.tokenized.de-en --destdir data-bin/toy
python train.py --config configs/smoke.yaml --device cpu
python generate.py --expdir logs/smoke --device cpu

# IWSLT'14 German → English.
python scripts/prepare_iwslt14.py
python preprocess.py --textdir data/iwslt14.tokenized.de-en --destdir data-bin/iwslt14.tokenized.de-en
python train.py --config configs/iwslt14_deen.yaml
python generate.py --expdir logs/iwslt14_deen --sacrebleu
```

Both preparation scripts expect the official corpus archive, which is downloaded separately. The checkpoint and binary corpus readers assume trusted local artifacts. Never load untrusted PyTorch checkpoint or corpus files.

## Evaluation

Generation writes predictions and a BLEU JSON report under the experiment directory. Tokenized BLEU strips BPE; optional detokenized sacreBLEU is a separate score. Compare only results measured on the same split with matching tokenization, decoding settings, and checkpoint selection. The toy corpus is for checking the pipeline, not for comparing translation quality.

An interrupted run resumes from the last completed epoch. Extending `max_epoch` is supported; use a new `expname` for changes to the training recipe. The [tests](tests/) cover masks, causal attention, cached decoding, beam search, loss, BLEU, checkpoint handling, and the toy pipeline.

## References

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), NeurIPS 2017.
- [fairseq translation examples](https://github.com/facebookresearch/fairseq/tree/main/examples/translation) for the IWSLT'14 data and training recipe. See [Third-party notices](THIRD_PARTY_NOTICES.md) for the adapted preprocessing script.
