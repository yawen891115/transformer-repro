"""End-to-end checks on a tiny generated corpus (no download, about a minute on a CPU)."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args, cwd=ROOT):
    subprocess.run([sys.executable, *map(str, args)], cwd=cwd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def test_prepare_preprocess_train_generate(tmp_path):
    tgz = tmp_path / "orig" / "de-en.tgz"
    run("scripts/make_toy_corpus.py", "--out", tgz, "--talks", "20", "--per-talk", "60",
        "--test-per-set", "20")
    text = tmp_path / "iwslt14.tokenized.de-en"
    run("scripts/prepare_iwslt14.py", "--orig", tgz.parent, "--out", text)

    n_train = sum(1 for _ in open(text / "train.de", encoding="utf-8"))
    n_valid = sum(1 for _ in open(text / "valid.de", encoding="utf-8"))
    n_test = sum(1 for _ in open(text / "test.en", encoding="utf-8"))
    total = 20 * (60 + 2)                       # sentences + title + description per talk
    assert n_valid == total // 23 and n_train == total - n_valid and n_test == 5 * 20

    data_bin = tmp_path / "data-bin"
    run("preprocess.py", "--textdir", text, "--destdir", data_bin)
    run("train.py", "--config", "configs/smoke.yaml", "--set", f"data_bin={data_bin}",
        f"logdir={tmp_path / 'logs'}", "max_epoch=2", "bleu_start_epoch=2", "bleu_every=1",
        "d_model=32", "d_ff=64", "enc_layers=1", "dec_layers=1")
    expdir = tmp_path / "logs" / "smoke"
    assert (expdir / "checkpoint_best.pt").exists() and (expdir / "checkpoint_last.pt").exists()

    run("generate.py", "--expdir", expdir, "--beam", "2")
    result = json.loads((expdir / "bleu-test.json").read_text())
    assert 0.0 <= result["tokenized_bleu"] <= 100.0
    lines = (expdir / "generate-test.txt").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3 * n_test


def test_resume_continues_from_the_last_epoch(tmp_path):
    tgz = tmp_path / "orig" / "de-en.tgz"
    run("scripts/make_toy_corpus.py", "--out", tgz, "--talks", "5", "--per-talk", "40",
        "--test-per-set", "5")
    text = tmp_path / "text"
    run("scripts/prepare_iwslt14.py", "--orig", tgz.parent, "--out", text)
    run("preprocess.py", "--textdir", text, "--destdir", tmp_path / "bin")
    common = ["--config", "configs/smoke.yaml", "--set", f"data_bin={tmp_path / 'bin'}",
              f"logdir={tmp_path / 'logs'}", "bleu_every=0", "d_model=32", "d_ff=64",
              "enc_layers=1", "dec_layers=1"]
    run("train.py", *common, "max_epoch=1")
    run("train.py", *common, "max_epoch=2")      # same command, larger budget: resumes
    import torch
    ckpt = torch.load(tmp_path / "logs" / "smoke" / "checkpoint_last.pt", weights_only=False)
    assert ckpt["epoch"] == 2
    assert (tmp_path / "logs" / "smoke" / "checkpoint1.pt").exists()
    # With BLEU switched off the best checkpoint is chosen by validation loss.
    assert (tmp_path / "logs" / "smoke" / "checkpoint_best.pt").exists()
