"""Pre-train SwinUNETR or UNETR on binary tumour masks from BraTS 2021."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import json
import math
from multiprocessing import get_context
import os
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from hyps import data
from hyps.config import validate_model
from hyps.engine import amp_ctx, runtime_device, save_checkpoint, seed_all, seed_worker, tile_predict
from hyps.metrics import dice
from hyps.models import build


def brats_cases(root, case_list):
    if case_list:
        text = case_list.read_text(encoding="utf-8")
        subjects = sorted(set(re.findall(r"BraTS2021_\d{5}(?!\d)", text)))
    else:
        subjects = sorted(path.name for path in root.glob("BraTS2021_*")
                          if path.is_dir() and re.fullmatch(r"BraTS2021_\d{5}", path.name))
    cases = []
    for subject in subjects:
        image = root / subject / f"{subject}_t1ce.nii.gz"
        label = root / subject / f"{subject}_seg.nii.gz"
        if image.is_file() and label.is_file():
            cases.append(dict(id=subject, image=image, label=label))
        elif case_list or label.is_file():
            raise ValueError(f"Missing T1ce image or mask for {subject}")
    if len(cases) < 2:
        raise ValueError("Provide at least two BraTS cases with T1ce images and segmentation masks")
    return [case for index, case in enumerate(cases) if index % 20], cases[::20]


def cache_files(case, cache_dir):
    key = data.cache_key(case["image"], case["label"])
    return cache_dir / f"brats_{key}_img.npy", cache_dir / f"brats_{key}_lab.npy"


def cache_case(case, cache_dir):
    image_file, label_file = cache_files(case, cache_dir)
    if image_file.is_file() and label_file.is_file():
        return
    image, label, _ = data.load_case(case, normalize=False)
    brain = np.argwhere(image > 0)
    if not len(brain):
        raise ValueError(f"No positive T1ce voxels for {case['id']}")
    lo = np.maximum(brain.min(axis=0) - 2, 0)
    hi = np.minimum(brain.max(axis=0) + 3, image.shape)
    region = tuple(slice(start, stop) for start, stop in zip(lo, hi))
    image = data.zscore_positive(image[region]).astype(np.float16)
    data.save_array(image_file, image)
    data.save_array(label_file, label[region])


class BratsPatches(Dataset):
    def __init__(self, files, patch):
        self.files = files
        self.patch = patch

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image_file, label_file = self.files[index]
        patch = self.patch
        image = data.pad_to_patch(np.load(image_file, mmap_mode="r"), patch)
        label = data.pad_to_patch(np.load(label_file, mmap_mode="r"), patch)
        if np.random.rand() < 0.5 and label.any():
            foreground = np.argwhere(label)
            center = foreground[np.random.randint(len(foreground))]
            starts = np.clip(center - patch // 2, 0, np.asarray(label.shape) - patch)
        else:
            starts = [np.random.randint(length - patch + 1) for length in label.shape]
        region = tuple(slice(start, start + patch) for start in starts)
        return data.augment_patch(image[region].astype(np.float32), label[region], 0.1)


def dice_ce(logits, label):
    logits = logits.float()
    prob = torch.softmax(logits, dim=1)[:, 1]
    foreground = label == 1
    soft_dice = (2 * (prob * foreground).sum() + 1e-5) / (prob.sum() + foreground.sum() + 1e-5)
    return F.cross_entropy(logits, label) + 1 - soft_dice


@torch.no_grad()
def validate(model, files, device, patch, amp):
    model.eval()
    scores = []
    for image_file, label_file in files:
        image = torch.from_numpy(np.load(image_file).astype(np.float32)).to(device)[None, None]
        prob = tile_predict(model, image, patch, amp, stride=patch // 2)
        prediction = (prob[0, 1] > prob[0, 0]).cpu().numpy()
        scores.append(dice(prediction, np.load(label_file)))
    return float(np.mean(scores))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brats-dir", type=Path, required=True)
    parser.add_argument("--case-list", type=Path, help="text or JSON containing source cohort IDs")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--arch", choices=["swinunetr", "unetr"], default="swinunetr")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, help="default 2e-4 for SwinUNETR and 1e-4 for UNETR")
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--batch", type=int, required=True, help="samples per GPU per forward pass")
    parser.add_argument("--accum", type=int, default=1, help="microbatches per optimizer step")
    parser.add_argument("--workers", type=int, required=True, help="data-loading workers per process")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--grad-ckpt", action="store_true", help="enable activation checkpointing for SwinUNETR")
    parser.add_argument("--cache-workers", type=int, default=4)
    parser.add_argument("--patch", type=int, default=128)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--amp", choices=["bf16", "fp16", "none"])
    args = parser.parse_args()
    if min(args.epochs, args.batch, args.accum, args.eval_every) < 1:
        parser.error("Epochs, batch size, accumulation and evaluation interval must be positive")
    if min(args.workers, args.cache_workers, args.warmup) < 0:
        parser.error("Worker counts and warm-up epochs must be non-negative")
    try:
        validate_model(dict(arch=args.arch, rank=1, scale=1.0), args.patch)
    except ValueError as error:
        parser.error(str(error))
    if args.lr is None:
        args.lr = 2e-4 if args.arch == "swinunetr" else 1e-4
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.wd) or args.wd < 0:
        parser.error("Learning rate must be positive and weight decay must be non-negative")
    args.amp = args.amp or ("none" if args.device == "cpu" else "bf16")
    if args.device == "cpu" and args.amp != "none":
        parser.error("--device cpu requires --amp none")
    return args


def pretrain(args, device, rank=0, world=1):
    on_cuda = device.type == "cuda"
    out_dir, cache_dir = args.out_dir, args.cache_dir
    last, final = out_dir / "model_last.pth", out_dir / "model_final.pth"
    if final.is_file() and not last.is_file():
        print(f"Pre-training already completed: {final}", flush=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    train, val = brats_cases(args.brats_dir, args.case_list)
    train_files = [cache_files(case, cache_dir) for case in train]
    val_files = [cache_files(case, cache_dir) for case in val]
    pending = [case for index, case in enumerate(train + val)
               if index % world == rank and not all(path.is_file() for path in cache_files(case, cache_dir))]
    if pending and args.cache_workers > 0:
        with ProcessPoolExecutor(max_workers=args.cache_workers, mp_context=get_context("spawn")) as pool:
            for _ in pool.map(cache_case, pending, repeat(cache_dir), chunksize=4):
                pass
    else:
        for case in pending:
            cache_case(case, cache_dir)
    if world > 1:
        dist.barrier()
    split = {part: [case["id"] for case in cases] for part, cases in (("train", train), ("val", val))}
    completed = torch.load(last, map_location="cpu", weights_only=True) if last.is_file() else {}
    if completed.get("split", split) != split:
        raise ValueError("The source cohort differs from the saved pre-training split")
    if rank == 0:
        (out_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")

    seed_all(2024 + rank)
    model = build(args.arch, use_checkpoint=args.grad_ckpt).to(device)
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.ndim <= 1 or any(term in name for term in ("relative_position_bias_table", "position_embeddings", "norm")):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                                   {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "fp16")
    dataset = BratsPatches(train_files, args.patch)
    sampler = DistributedSampler(dataset, shuffle=True) if world > 1 else None
    loader = DataLoader(dataset, batch_size=args.batch, sampler=sampler, shuffle=sampler is None,
                        num_workers=args.workers, pin_memory=on_cuda, drop_last=True,
                        worker_init_fn=seed_worker, persistent_workers=args.workers > 0)
    steps = len(loader)
    if not steps:
        raise ValueError("The training split is smaller than one batch per process")
    settings = dict(arch=args.arch, patch=args.patch, amp=args.amp, batch=args.batch, accum=args.accum,
                    world=world, lr=args.lr, wd=args.wd, warmup=args.warmup, epochs=args.epochs)
    log_path = out_dir / "train_log.csv"
    start, log = 0, []
    if completed:
        if completed.get("settings", settings) != settings:
            raise ValueError("Pre-training settings differ from model_last.pth")
        model.load_state_dict(completed["state_dict"])
        optimizer.load_state_dict(completed["optim"])
        if completed.get("scaler"):
            scaler.load_state_dict(completed["scaler"])
        start = completed["epoch"] + 1
        log = completed.get("log", [])
        if "log" not in completed and log_path.is_file():
            log = pd.read_csv(log_path).iloc[:start].to_dict("records")
        if rank == 0:
            print(f"Resuming from epoch {start + 1}", flush=True)
    network = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index]) if world > 1 else model
    total, warmup = args.epochs * steps, args.warmup * steps
    for epoch in range(start, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        network.train()
        begin = time.perf_counter()
        losses = []
        optimizer.zero_grad(set_to_none=True)
        for index, (image, label) in enumerate(loader):
            step = epoch * steps + index
            if step < warmup:
                lr = args.lr * step / warmup
            else:
                lr = args.lr * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))
            for group in optimizer.param_groups:
                group["lr"] = lr
            image = image.to(device, non_blocking=on_cuda)
            label = label.to(device, non_blocking=on_cuda)
            with amp_ctx(args.amp):
                logits = network(image)
            loss = dice_ce(logits, label)
            group_size = min(args.accum, steps - (index // args.accum) * args.accum)
            scaler.scale(loss / group_size).backward()
            losses.append(loss.item())
            if (index + 1) % args.accum == 0 or index + 1 == steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        if rank == 0:
            elapsed = time.perf_counter() - begin
            val_dice = None
            if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
                val_dice = validate(model, val_files, device, args.patch, args.amp)
            log.append(dict(epoch=epoch, lr=lr, train_loss=float(np.mean(losses)),
                            val_dice=val_dice, epoch_seconds=elapsed))
            save_checkpoint(dict(epoch=epoch, state_dict=model.state_dict(), optim=optimizer.state_dict(),
                                 scaler=scaler.state_dict(), split=split, settings=settings, log=log), last)
            pd.DataFrame(log).to_csv(log_path, index=False)
            print(f"epoch {epoch + 1}/{args.epochs}  loss {log[-1]['train_loss']:.4f}  val Dice {val_dice}", flush=True)
        if world > 1:
            dist.barrier()
    if rank == 0:
        pd.DataFrame(log).to_csv(log_path, index=False)
        save_checkpoint(dict(epoch=args.epochs, state_dict=model.state_dict()), final)
        last.unlink(missing_ok=True)
        print(f"Pre-training complete: {final}", flush=True)


def main():
    args = parse_args()
    device = runtime_device(args.device, args.amp)
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        pretrain(args, device)
        return
    if device.type != "cuda" or not dist.is_nccl_available():
        raise RuntimeError("Multi-GPU pre-training requires CUDA and the NCCL backend")
    torch.cuda.set_device(int(local_rank))
    dist.init_process_group("nccl")
    try:
        pretrain(args, torch.device("cuda", int(local_rank)), dist.get_rank(), dist.get_world_size())
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
