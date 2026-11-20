"""Regression tests for problems that only show up on a GPU or with unusual input files."""

import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import torch

import make_toy_corpus
import prepare_iwslt14
from transformer.config import load_config
from transformer.data import Dictionary, ParallelData, read_lines
from transformer.model import Transformer
from transformer.utils import amp_context, load_checkpoint, save_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def test_checkpoints_are_read_onto_the_cpu(tmp_path, monkeypatch):
    """If a checkpoint were loaded with map_location='cuda', the saved RNG state would become a
    CUDA tensor and torch.set_rng_state would fail when resuming on a GPU."""
    model = Transformer(16, 16, 1, d_model=16, d_ff=32, n_heads=2, enc_layers=1, dec_layers=1)
    opt = torch.optim.AdamW(model.parameters())
    cfg = load_config(None)
    save_checkpoint(tmp_path / "ckpt.pt", model, opt, 3, 99, 12.5, cfg)
    real_load, seen = torch.load, []

    def checked_load(*args, **kwargs):
        seen.append(kwargs.get("map_location"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", checked_load)
    ckpt = load_checkpoint(tmp_path / "ckpt.pt", model, opt)
    assert seen == ["cpu"] and ckpt["epoch"] == 3 and ckpt["num_updates"] == 99


def test_only_newline_ends_a_line(tmp_path):
    """A stray carriage return inside a sentence must not split it in two (Perl/sed agree)."""
    f = tmp_path / "x.txt"
    f.write_bytes("ein\rsatz\nzwei\n".encode("utf-8"))
    assert read_lines(f) == ["ein\rsatz", "zwei"]


def test_misaligned_files_are_rejected(tmp_path):
    (tmp_path / "a.de").write_text("a\nb\n", encoding="utf-8")
    (tmp_path / "a.en").write_text("a\n", encoding="utf-8")
    d = Dictionary.build([tmp_path / "a.de"])
    with pytest.raises(ValueError):
        ParallelData.from_text(tmp_path / "a.de", tmp_path / "a.en", d, d)


def test_amp_setting():
    with amp_context("bf16", torch.device("cpu"))():           # bf16 only applies on a GPU
        pass
    with pytest.raises(ValueError):
        amp_context("fp16", torch.device("cpu"))
    with pytest.raises(ValueError):
        load_config(None, ["amp=fp16"])


@pytest.mark.filterwarnings("ignore::DeprecationWarning")   # newer Pythons warn on this path
def test_prepare_works_without_tarfile_filters(tmp_path, monkeypatch):
    """Python versions before the tarfile 'filter' backport must still unpack the archive."""
    tgz = tmp_path / "orig" / "de-en.tgz"
    monkeypatch.setattr(sys, "argv", ["make_toy_corpus.py", "--out", str(tgz), "--talks", "3",
                                      "--per-talk", "30", "--test-per-set", "4"])
    make_toy_corpus.main()
    monkeypatch.delattr(tarfile, "data_filter", raising=False)
    out = tmp_path / "prep"
    monkeypatch.setattr(sys, "argv", ["prepare_iwslt14.py", "--orig", str(tgz.parent),
                                      "--out", str(out)])
    prepare_iwslt14.main()
    assert len(read_lines(out / "test.de")) == 5 * 4


def test_generate_falls_back_to_the_last_checkpoint(tmp_path):
    def run(*args):
        subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    tgz = tmp_path / "orig" / "de-en.tgz"
    run("scripts/make_toy_corpus.py", "--out", tgz, "--talks", "4", "--per-talk", "30",
        "--test-per-set", "4")
    run("scripts/prepare_iwslt14.py", "--orig", tgz.parent, "--out", tmp_path / "text")
    run("preprocess.py", "--textdir", tmp_path / "text", "--destdir", tmp_path / "bin")
    run("train.py", "--config", "configs/smoke.yaml", "--set", f"data_bin={tmp_path / 'bin'}",
        f"logdir={tmp_path / 'logs'}", "max_epoch=1", "bleu_start_epoch=5", "d_model=32",
        "d_ff=64", "enc_layers=1", "dec_layers=1")
    expdir = tmp_path / "logs" / "smoke"
    assert not (expdir / "checkpoint_best.pt").exists()      # BLEU never ran, so no "best"
    run("generate.py", "--expdir", expdir, "--beam", "2")
    result = json.loads((expdir / "bleu-test.json").read_text())
    assert result["checkpoint"].endswith("checkpoint_last.pt")
