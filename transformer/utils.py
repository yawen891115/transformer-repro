"""Model construction, learning-rate schedule, checkpoints and small helpers."""

from __future__ import annotations

import contextlib
import random
from pathlib import Path

import numpy as np
import torch

from .config import TransformerConfig
from .data import Dictionary
from .model import Transformer


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dictionaries(cfg: TransformerConfig):
    data_bin = Path(cfg.data_bin)
    return (Dictionary.load(data_bin / f"dict.{cfg.src_lang}.txt"),
            Dictionary.load(data_bin / f"dict.{cfg.tgt_lang}.txt"))


def build_model(cfg: TransformerConfig, src_dict: Dictionary, tgt_dict: Dictionary) -> Transformer:
    return Transformer(len(src_dict), len(tgt_dict), src_dict.pad, cfg.d_model, cfg.d_ff,
                       cfg.n_heads, cfg.enc_layers, cfg.dec_layers, cfg.dropout,
                       cfg.attn_dropout, cfg.act_dropout, cfg.max_positions,
                       cfg.share_all_embeddings)


def inverse_sqrt_lr(step: int, peak_lr: float, warmup: int) -> float:
    """Linear warm-up from 0 to peak_lr over `warmup` updates, then peak_lr * sqrt(warmup / step).
    Same shape as the paper's  d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)  (Eq. 3),
    but parameterised by the peak value, as fairseq's `inverse_sqrt` scheduler is."""
    if step < warmup:
        return peak_lr * step / warmup
    return peak_lr * (warmup / step) ** 0.5


def amp_context(amp: str, device: torch.device):
    """Return a function that opens the mixed-precision context for each step:
    bfloat16 autocast when amp == "bf16" on a CUDA GPU that supports it, otherwise fp32."""
    if amp not in ("none", "bf16"):
        raise ValueError(f"amp must be 'none' or 'bf16', got {amp!r}")
    if amp == "bf16" and device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return lambda: torch.autocast("cuda", dtype=torch.bfloat16)
        print("warning: this GPU does not support bfloat16; running in fp32")
    return contextlib.nullcontext


def save_checkpoint(path: Path, model, optimizer, epoch: int, num_updates: int,
                    best_bleu: float, cfg: TransformerConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    rng = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
           "python": random.getstate()}
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": epoch, "num_updates": num_updates, "best_bleu": best_bleu,
                "rng": rng}, tmp)
    tmp.replace(path)                                   # atomic rename


def load_checkpoint(path: Path, model, optimizer=None) -> dict:
    """Load weights; with an optimizer also restore its state and the random-number states.

    Everything is read onto the CPU first: the random-number states must stay CPU byte
    tensors, and load_state_dict copies the weights to whatever device the model is on.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        torch.set_rng_state(ckpt["rng"]["torch"])
        np.random.set_state(ckpt["rng"]["numpy"])
        random.setstate(ckpt["rng"]["python"])
        if "cuda" in ckpt["rng"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ckpt["rng"]["cuda"])
    return ckpt
