"""Train a Transformer on IWSLT'14 De->En (fairseq's IWSLT example, re-implemented).

    python train.py --config configs/iwslt14_deen.yaml
    python train.py --config configs/iwslt14_deen.yaml --set amp=bf16 max_epoch=50

Every epoch: train, compute validation loss, and (from bleu_start_epoch on) validation BLEU
with beam search. Writes checkpoint_last.pt (used to resume), checkpoint_best.pt (best
validation BLEU) and checkpoint<epoch>.pt (last few, for averaging) to logs/<expname>/.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from transformer.bleu import corpus_bleu
from transformer.config import load_config, save_config
from transformer.data import ParallelData, collate, make_batches
from transformer.loss import label_smoothed_nll_loss
from transformer.search import beam_search
from transformer.utils import (amp_context, build_model, inverse_sqrt_lr, load_checkpoint,
                               load_dictionaries, pick_device, save_checkpoint, seed_everything)

LN2 = math.log(2)


def batch_to(batch: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()}


@torch.no_grad()
def validation_loss(model, data, cfg, pad, bos, device, autocast) -> tuple[float, float]:
    """Label-smoothed loss and plain NLL per target token, in base 2 (as fairseq logs them)."""
    model.eval()
    tot_loss = tot_nll = tot_tok = 0.0
    for idx in make_batches(data, cfg.max_tokens, shuffle=False):
        b = batch_to(collate(data, idx, pad, bos), device)
        with autocast():
            logits = model(b["src_tokens"], b["prev_tokens"])
        lprobs = torch.log_softmax(logits.float(), dim=-1).view(-1, logits.shape[-1])
        loss, nll = label_smoothed_nll_loss(lprobs, b["target"].view(-1), cfg.label_smoothing, pad)
        tot_loss += loss.item()
        tot_nll += nll.item()
        tot_tok += b["ntokens"]
    return tot_loss / tot_tok / LN2, tot_nll / tot_tok / LN2


@torch.no_grad()
def validation_bleu(model, data, tgt_dict, cfg, device) -> dict:
    """Beam search on the validation set; BLEU on tokenised text with BPE removed."""
    model.eval()
    hyps, refs = [""] * len(data), [""] * len(data)
    batches = make_batches(data, cfg.max_tokens, shuffle=False,
                           max_sentences=cfg.max_sentences_decode)
    for idx in batches:
        b = batch_to(collate(data, idx, tgt_dict.pad, tgt_dict.bos), device)
        outs = beam_search(model, b["src_tokens"], cfg.beam, tgt_dict.bos, tgt_dict.eos,
                           tgt_dict.pad, cfg.valid_max_len_a, cfg.valid_max_len_b, cfg.lenpen)
        for i, out in zip(idx, outs):
            hyps[i] = tgt_dict.decode(out, remove_bpe=True)
            refs[i] = tgt_dict.decode(data.tgt[i].tolist(), remove_bpe=True, escape_unk=True)
    return corpus_bleu(hyps, refs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    device = pick_device(args.device)
    seed_everything(cfg.seed)
    expdir = cfg.expdir
    config_record = expdir / "config.yaml"
    last = expdir / "checkpoint_last.pt"
    if last.exists():
        if not config_record.is_file():
            raise RuntimeError(f"Missing saved config for {last}; use a new expname.")
        old_cfg = load_config(str(config_record))
        tunable = {"max_epoch", "log_every", "keep_epoch_ckpts"}
        changed = sorted(f.name for f in dataclasses.fields(cfg)
                         if f.name not in tunable and getattr(cfg, f.name) != getattr(old_cfg, f.name))
        if changed:
            raise ValueError(f"Resume config differs in {', '.join(changed)}; use a new expname.")

    src_dict, tgt_dict = load_dictionaries(cfg)
    train = ParallelData.load(f"{cfg.data_bin}/train.pt")
    valid = ParallelData.load(f"{cfg.data_bin}/valid.pt")
    pad, bos = tgt_dict.pad, tgt_dict.bos
    print(f"device={device}  train={len(train)} pairs  valid={len(valid)} pairs  "
          f"vocab {cfg.src_lang}={len(src_dict)} {cfg.tgt_lang}={len(tgt_dict)}")

    model = build_model(cfg, src_dict, tgt_dict).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=cfg.adam_betas,
                                  eps=cfg.adam_eps, weight_decay=cfg.weight_decay)

    # best_bleu holds the best validation BLEU (or -loss when BLEU is off); -inf = none yet
    epoch, num_updates, best_bleu = 0, 0, float("-inf")
    if last.exists():
        ckpt = load_checkpoint(last, model, optimizer)
        epoch, num_updates, best_bleu = ckpt["epoch"], ckpt["num_updates"], ckpt["best_bleu"]
        print(f"resumed from {last} (epoch {epoch}, {num_updates} updates)")

    save_config(cfg, config_record)

    autocast = amp_context(cfg.amp, device)
    writer = SummaryWriter(str(expdir / "tensorboard"))

    while epoch < cfg.max_epoch:
        epoch += 1
        model.train()
        batches = make_batches(train, cfg.max_tokens, shuffle=True, seed=cfg.seed + epoch)
        t0, log_loss, log_nll, log_tok = time.time(), 0.0, 0.0, 0
        pbar = tqdm(batches, desc=f"epoch {epoch}", dynamic_ncols=True)
        for idx in pbar:
            b = batch_to(collate(train, idx, pad, bos), device)
            lr = inverse_sqrt_lr(num_updates + 1, cfg.lr, cfg.warmup_updates)
            for group in optimizer.param_groups:
                group["lr"] = lr

            with autocast():
                logits = model(b["src_tokens"], b["prev_tokens"])
            lprobs = torch.log_softmax(logits.float(), dim=-1).view(-1, logits.shape[-1])
            loss, nll = label_smoothed_nll_loss(lprobs, b["target"].view(-1),
                                                cfg.label_smoothing, pad)
            optimizer.zero_grad(set_to_none=True)
            (loss / b["ntokens"]).backward()            # mean over target tokens
            if cfg.clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
            optimizer.step()
            num_updates += 1

            log_loss, log_nll, log_tok = log_loss + loss.item(), log_nll + nll.item(), \
                log_tok + b["ntokens"]
            if num_updates % cfg.log_every == 0:
                l2, n2 = log_loss / log_tok / LN2, log_nll / log_tok / LN2
                pbar.set_postfix(loss=f"{l2:.3f}", ppl=f"{2 ** n2:.2f}", lr=f"{lr:.2e}")
                writer.add_scalar("train/loss", l2, num_updates)
                writer.add_scalar("train/ppl", 2 ** n2, num_updates)
                writer.add_scalar("train/lr", lr, num_updates)
                writer.add_scalar("train/tokens_per_sec", log_tok / (time.time() - t0), num_updates)
                t0, log_loss, log_nll, log_tok = time.time(), 0.0, 0.0, 0

        v_loss, v_nll = validation_loss(model, valid, cfg, pad, bos, device, autocast)
        writer.add_scalar("valid/loss", v_loss, epoch)
        writer.add_scalar("valid/ppl", 2 ** v_nll, epoch)
        msg = f"epoch {epoch}: valid loss {v_loss:.3f}  nll {v_nll:.3f}  ppl {2 ** v_nll:.2f}"

        improved = False
        if cfg.bleu_every > 0 and epoch >= cfg.bleu_start_epoch and epoch % cfg.bleu_every == 0:
            bleu = validation_bleu(model, valid, tgt_dict, cfg, device)["bleu"]
            writer.add_scalar("valid/bleu", bleu, epoch)
            msg += f"  BLEU {bleu:.2f}"
            if bleu > best_bleu:
                best_bleu, improved = bleu, True
        elif cfg.bleu_every == 0 and -v_loss > best_bleu:     # no BLEU: select by valid loss
            best_bleu, improved = -v_loss, True
        print(msg + ("  (best)" if improved else ""))

        save_checkpoint(expdir / "checkpoint_last.pt", model, optimizer, epoch, num_updates,
                        best_bleu, cfg)
        save_checkpoint(expdir / f"checkpoint{epoch}.pt", model, optimizer, epoch, num_updates,
                        best_bleu, cfg)
        old = expdir / f"checkpoint{epoch - cfg.keep_epoch_ckpts}.pt"
        if old.exists():
            old.unlink()
        if improved:
            save_checkpoint(expdir / "checkpoint_best.pt", model, optimizer, epoch, num_updates,
                            best_bleu, cfg)
    writer.close()
    print(f"done. Next: python generate.py --expdir {expdir}")


if __name__ == "__main__":
    main()
