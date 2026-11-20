"""Experiment settings.

Every run is described by one `TransformerConfig`. Values come from a YAML file in
`configs/` and can be overridden on the command line, for example:

    python train.py --config configs/iwslt14_deen.yaml --set max_epoch=50 amp=bf16
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class TransformerConfig:
    # ----- experiment -----
    expname: str = "iwslt14_deen"
    logdir: str = "logs"
    seed: int = 1
    data_bin: str = "data-bin/iwslt14.tokenized.de-en"   # written by preprocess.py
    src_lang: str = "de"
    tgt_lang: str = "en"

    # ----- model: fairseq transformer_iwslt_de_en -----
    d_model: int = 512
    d_ff: int = 1024
    n_heads: int = 4
    enc_layers: int = 6
    dec_layers: int = 6
    dropout: float = 0.3
    attn_dropout: float = 0.0
    act_dropout: float = 0.0
    max_positions: int = 1024
    share_all_embeddings: bool = False     # needs preprocess.py --joined-dictionary

    # ----- optimisation (fairseq IWSLT'14 example command) -----
    max_tokens: int = 4096                 # tokens per batch, padding included
    lr: float = 5e-4                       # peak learning rate
    warmup_updates: int = 4000             # inverse square-root schedule
    adam_betas: tuple = (0.9, 0.98)
    adam_eps: float = 1e-8
    weight_decay: float = 1e-4             # decoupled (AdamW-style), as in fairseq's Adam
    clip_norm: float = 0.0                 # 0 = no gradient clipping
    label_smoothing: float = 0.1
    max_epoch: int = 60
    amp: str = "none"                      # none | bf16

    # ----- validation / decoding -----
    beam: int = 5
    lenpen: float = 1.0
    valid_max_len_a: float = 1.2           # validation BLEU: max length 1.2 * src + 10
    valid_max_len_b: int = 10
    bleu_every: int = 1                    # validation BLEU every k epochs (0 = never)
    bleu_start_epoch: int = 10             # ... starting at this epoch (early epochs are slow)
    max_sentences_decode: int = 128

    # ----- logging / checkpoints -----
    log_every: int = 100                   # updates
    keep_epoch_ckpts: int = 10             # keep the last k epoch checkpoints (for averaging)

    @property
    def expdir(self) -> Path:
        return Path(self.logdir) / self.expname


def load_config(path: str | None, overrides: list[str] | None = None) -> TransformerConfig:
    values: dict = {}
    if path is not None:
        with open(path) as f:
            values.update(yaml.safe_load(f) or {})
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key=value, got {item!r}")
        key, text = item.split("=", 1)
        values[key.strip()] = yaml.safe_load(text)
    known = {f.name for f in dataclasses.fields(TransformerConfig)}
    unknown = set(values) - known
    if unknown:
        raise KeyError(f"unknown config keys: {sorted(unknown)}")
    if "adam_betas" in values:
        values["adam_betas"] = tuple(values["adam_betas"])
    cfg = TransformerConfig(**values)
    if cfg.amp not in ("none", "bf16"):
        raise ValueError(f"amp must be 'none' or 'bf16', got {cfg.amp!r}")
    return cfg


def save_config(cfg: TransformerConfig, path: Path) -> None:
    data = dataclasses.asdict(cfg)
    data["adam_betas"] = list(cfg.adam_betas)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".yaml.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    tmp.replace(path)
