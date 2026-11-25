"""Load the experiment configuration."""
import math
import os
from pathlib import Path

import yaml

REQUIRED = {
    "paths": ["split_file", "pretrained", "output_dir", "cache_dir"],
    "device": ["gpu", "num_workers", "micro_batch", "amp"],
    "model": ["arch", "rank", "scale"],
    "train": ["epochs", "batch_size", "lr", "poly_power", "weight_decay", "amsgrad", "patch", "intensity"],
    "inference": ["ensemble_last", "min_component"],
}


def load_config(path, require=REQUIRED):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise SystemExit(f"Expected a YAML mapping in {path}")
    for section in REQUIRED:
        if cfg.get(section) is None:
            cfg[section] = {}
        if not isinstance(cfg[section], dict):
            raise SystemExit(f"{section} must be a YAML mapping")
    missing = [f"{section}.{key}" for section, keys in require.items() for key in keys
               if cfg[section].get(key) in (None, "")]
    if missing:
        raise SystemExit(f"Please fill in the following fields of {path}: " + ", ".join(missing))
    device = cfg["device"]
    gpu = device.get("gpu")
    if gpu is not None and gpu != "cpu" and (type(gpu) is not int or gpu < 0):
        raise SystemExit("device.gpu must be a non-negative GPU index or 'cpu'")
    amp = device.get("amp", "none")
    if amp not in ("bf16", "fp16", "none"):
        raise SystemExit("device.amp must be one of: bf16, fp16, none")
    if gpu == "cpu" and amp != "none":
        raise SystemExit("device.amp must be 'none' for CPU runs")
    for section, key, minimum in (
        ("device", "num_workers", 0), ("device", "micro_batch", 1),
        ("model", "rank", 1), ("train", "epochs", 1), ("train", "batch_size", 1),
        ("train", "patch", 1), ("inference", "ensemble_last", 1), ("inference", "min_component", 0),
    ):
        value = cfg[section].get(key)
        if value is not None and (type(value) is not int or value < minimum):
            raise SystemExit(f"{section}.{key} must be an integer >= {minimum}")
    if "model" in require:
        try:
            validate_model(cfg["model"], cfg["train"].get("patch") or 128)
        except ValueError as error:
            raise SystemExit(str(error)) from error
    settings = cfg["train"]
    for key in ("lr", "poly_power", "weight_decay", "intensity"):
        value = settings.get(key)
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
            raise SystemExit(f"train.{key} must be a finite non-negative number")
    seed = settings.get("seed", 0)
    if type(seed) is not int or not 0 <= seed < 2 ** 32:
        raise SystemExit("train.seed must be an integer from 0 to 2**32 - 1")
    return cfg


def validate_model(model, patch):
    arch = model["arch"]
    if arch not in ("swinunetr", "unetr"):
        raise ValueError("model.arch must be 'swinunetr' or 'unetr'")
    if type(model["rank"]) is not int or model["rank"] < 1:
        raise ValueError("model.rank must be a positive integer")
    scale = model["scale"]
    if type(scale) not in (int, float) or not math.isfinite(scale):
        raise ValueError("model.scale must be a finite number")
    stages = model.get("layers")
    if stages and (arch != "swinunetr" or not isinstance(stages, list)
                   or any(stage not in ("layers1", "layers2", "layers3", "layers4") for stage in stages)):
        raise ValueError("model.layers must be a list of SwinUNETR stages (layers1 through layers4)")
    if model.get("train", "decoder") not in ("decoder", "head"):
        raise ValueError("model.train must be 'decoder' or 'head'")
    if type(patch) is not int:
        raise ValueError("train.patch must be an integer")
    if arch == "unetr" and patch != 128:
        raise ValueError("UNETR requires train.patch: 128")
    if arch == "swinunetr" and (patch < 64 or patch % 32 != 0):
        raise ValueError("SwinUNETR requires train.patch to be a multiple of 32 and at least 64")


def select_gpu(cfg):
    """Must be called before torch initializes CUDA."""
    gpu = cfg["device"]["gpu"]
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if gpu == "cpu" else str(gpu)
