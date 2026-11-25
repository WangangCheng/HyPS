"""Fine-tuning, tiled inference and evaluation."""
from contextlib import nullcontext
import hashlib
import json
import random
import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import data as D
from . import metrics as M
from .hyps_linear import apply_hyps, count_params, trainable_state
from .models import build


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def seed_worker(_worker_id):
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


def amp_ctx(amp):
    if amp == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if amp == "fp16":
        return torch.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def runtime_device(gpu, amp):
    if gpu == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install a CUDA build of PyTorch or use device.gpu: cpu with amp: none.")
    if amp == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU does not support bf16. Use fp16 or none.")
    return torch.device("cuda")


def checkpoint_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(checkpoint, path):
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        pending = Path(handle.name)
    try:
        torch.save(checkpoint, pending)
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def dice_loss(logits, target):
    """Sum the class-wise soft Dice losses over the current microbatch (Eq. 5)."""
    prob = torch.softmax(logits.float(), dim=1)
    loss = 0.0
    for c in (0, 1):
        p = prob[:, c]
        t = (target == c).float()
        loss = loss + 1.0 - 2.0 * (p * t).sum() / (p.sum() + t.sum() + 1e-5)
    return loss


def build_hyps_model(mcfg, pretrained):
    model = build(mcfg["arch"], pretrained=pretrained)
    n = apply_hyps(model, mcfg["arch"], r=mcfg["rank"], s=mcfg["scale"], layers=mcfg.get("layers") or None,
                   train=mcfg.get("train", "decoder"))
    return model, n


@torch.no_grad()
def tile_predict(model, x, patch=128, amp="bf16", stride=None):
    """Average softmax probabilities over overlapping, boundary-aligned tiles."""
    height, width, depth = x.shape[-3:]
    stride = patch if stride is None else stride
    out = torch.zeros((1, 2, height, width, depth), device=x.device)
    count = torch.zeros((1, 1, height, width, depth), device=x.device)
    starts = [sorted({min(i, max(length - patch, 0)) for i in range(0, length, stride)})
              for length in (height, width, depth)]
    for h in starts[0]:
        for w in starts[1]:
            for z in starts[2]:
                region = (slice(h, h + patch), slice(w, w + patch), slice(z, z + patch))
                tile = x[(...,) + region]
                dh, dw, dz = tile.shape[-3:]
                padding = (0, patch - dz, 0, patch - dw, 0, patch - dh)
                if any(padding):
                    tile = torch.nn.functional.pad(tile, padding)
                with amp_ctx(amp):
                    prob = torch.softmax(model(tile).float(), dim=1)
                out[(...,) + region] += prob[..., :dh, :dw, :dz]
                count[(...,) + region] += 1
    return out / count


@torch.no_grad()
def evaluate(model, snapshots, cases, out_dir, set_name, amp, patch=128, min_component=1000, cache_dir=None,
             with_label=True, save_predictions=False, num_workers=0):
    """Evaluate the ensemble and optionally write prediction masks."""
    if not snapshots:
        raise ValueError("At least one HyPS snapshot is required")
    trainable = {name: value.shape for name, value in model.named_parameters() if value.requires_grad}
    for i, snapshot in enumerate(snapshots, start=1):
        if snapshot.keys() != trainable.keys() or any(value.shape != trainable[name]
                                                       for name, value in snapshot.items()):
            raise ValueError(f"HyPS snapshot {i} does not match the model's trainable parameters")
    device = next(model.parameters()).device
    model.eval()
    ds = D.EvalSet(cases, cache_dir=cache_dir, with_label=with_label)
    dl = DataLoader(ds, batch_size=1, num_workers=num_workers, collate_fn=D.collate_eval)
    pred_dir = Path(out_dir) / f"pred_{set_name}"
    if save_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sid, img, label, affine in dl:
        x = img[None].to(device)
        t0 = time.perf_counter()
        prob = None
        for snap in snapshots:
            model.load_state_dict(snap, strict=False)
            p = tile_predict(model, x, patch=patch, amp=amp)
            prob = p if prob is None else prob + p
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0
        pred = (prob[0, 1] > prob[0, 0]).cpu().numpy().astype(np.uint8)
        pred = M.postprocess(pred, min_component)
        r = dict(subject=sid, infer_seconds=dt)
        r.update(M.evaluate_case(pred, label, affine))
        rows.append(r)
        if save_predictions:
            mask = nib.Nifti1Image(pred, affine)
            mask.header.set_xyzt_units("mm")
            nib.save(mask, pred_dir / f"{sid}_pred.nii.gz")
    df = pd.DataFrame(rows)
    df.to_csv(Path(out_dir) / f"metrics_{set_name}.csv", index=False)
    return df


def train(cfg):
    paths, hardware, model_cfg = cfg["paths"], cfg["device"], cfg["model"]
    settings, inference = cfg["train"], cfg["inference"]
    amp = hardware["amp"]
    device = runtime_device(hardware["gpu"], amp)
    on_cuda = device.type == "cuda"
    out = Path(paths["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    seed_all(settings.get("seed", 0))
    torch.backends.cudnn.benchmark = on_cuda

    split = D.load_split(paths["split_file"], paths.get("data_root"))
    dataset = D.TrainSet(split["train"], paths["cache_dir"], patch=settings["patch"], intensity=settings["intensity"])
    digest = checkpoint_digest(paths["pretrained"])
    model, n_wrapped = build_hyps_model(model_cfg, paths["pretrained"])
    model.to(device)
    params = count_params(model)
    print(f"HyPS applied to {n_wrapped} linear layers; trainable parameters: "
          f"encoder {params['trainable_encoder'] / 1e6:.2f} M, total {params['trainable_total'] / 1e6:.2f} M", flush=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=settings["lr"], weight_decay=settings["weight_decay"],
                                 amsgrad=settings["amsgrad"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp == "fp16")
    batch_size, micro_batch = settings["batch_size"], hardware["micro_batch"]
    workers = hardware["num_workers"]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=len(dataset) >= batch_size,
                        num_workers=workers, pin_memory=on_cuda, worker_init_fn=seed_worker,
                        persistent_workers=workers > 0)
    epochs, keep = settings["epochs"], inference["ensemble_last"]
    snapshots, log = [], []
    if on_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    t_train = time.perf_counter()
    for epoch in range(epochs):
        lr = settings["lr"] * (1 - epoch / epochs) ** settings["poly_power"]
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        t0 = time.perf_counter()
        losses = []
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for xs, ys in zip(x.split(micro_batch), y.split(micro_batch)):
                xs = xs.to(device, non_blocking=on_cuda)
                ys = ys.to(device, non_blocking=on_cuda)
                with amp_ctx(amp):
                    logits = model(xs)
                loss = dice_loss(logits, ys) * (xs.shape[0] / x.shape[0])
                scaler.scale(loss).backward()
                total_loss += loss.item()
            scaler.step(optimizer)
            scaler.update()
            losses.append(total_loss)
        log.append(dict(epoch=epoch, lr=lr, loss=float(np.mean(losses)), seconds=time.perf_counter() - t0))
        if epoch >= epochs - keep:
            snapshots.append(trainable_state(model))
        if epoch % 50 == 0 or epoch == epochs - 1:
            pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)
            print(f"epoch {epoch + 1}/{epochs}  loss {log[-1]['loss']:.4f}  lr {lr:.2e}  "
                  f"{log[-1]['seconds']:.1f}s", flush=True)
    summary = dict(train_seconds=time.perf_counter() - t_train,
                   peak_mem_gb_train=torch.cuda.max_memory_allocated(device) / 2 ** 30 if on_cuda else None,
                   n_train=len(split["train"]), **params)
    save_checkpoint(dict(model=model_cfg, snapshots=snapshots, pretrained_sha256=digest), out / "hyps_weights.pt")

    for set_name in ("val", "test"):
        cases = split.get(set_name) or []
        if not cases:
            continue
        df = evaluate(model, snapshots, cases, out, set_name, amp, patch=settings["patch"],
                      min_component=inference["min_component"], cache_dir=paths["cache_dir"],
                      save_predictions=inference.get("save_predictions", False), num_workers=workers)
        if "dice" in df:
            summary[f"dice_{set_name}"] = float(df["dice"].mean())
            defined_hd95 = df["hd95"].dropna()
            summary[f"n_hd95_{set_name}"] = len(defined_hd95)
            summary[f"hd95_{set_name}"] = float(defined_hd95.mean()) if len(defined_hd95) else None
            print(f"{set_name}: Dice {df['dice'].mean():.2f}  HD95 {df['hd95'].mean():.3f}", flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary
