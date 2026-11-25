"""Segment preprocessed NIfTI images with a trained HyPS ensemble."""
import argparse
from pathlib import Path

from hyps.config import load_config, select_gpu, validate_model

NEEDED = {"paths": ["pretrained"], "device": ["gpu", "amp"], "train": ["patch"], "inference": ["min_component"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True, help="hyps_weights.pt written by train.py")
    parser.add_argument("--input", required=True, type=Path, help="a .nii or .nii.gz file, or a folder of NIfTI files")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ensemble", type=int, help="number of saved epochs to average (default: all)")
    args = parser.parse_args()
    if args.ensemble is not None and args.ensemble < 1:
        parser.error("--ensemble must be a positive integer")
    if args.input.is_dir():
        files = sorted(args.input.iterdir())
    else:
        files = [args.input]
    files = [path for path in files if path.is_file() and path.name.endswith((".nii", ".nii.gz"))]
    if not files:
        parser.error(f"No NIfTI images found at {args.input}")
    cfg = load_config(args.config, require=NEEDED)
    select_gpu(cfg)

    import torch
    from hyps.data import nifti_id, validate_id
    from hyps.engine import build_hyps_model, checkpoint_digest, evaluate, runtime_device

    ids = [validate_id(nifti_id(path)) for path in files]
    if len({subject.casefold() for subject in ids}) != len(ids):
        parser.error("Input NIfTI files share the same case ID")
    cases = [dict(id=subject, image=path) for subject, path in zip(ids, files)]
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=True)
    snapshots = ckpt["snapshots"]
    if not snapshots or (args.ensemble is not None and args.ensemble > len(snapshots)):
        parser.error(f"The checkpoint has {len(snapshots)} snapshots")
    if args.ensemble is not None:
        snapshots = snapshots[-args.ensemble:]
    pretrained = cfg["paths"]["pretrained"]
    expected_digest = ckpt.get("pretrained_sha256")
    if expected_digest and checkpoint_digest(pretrained) != expected_digest:
        parser.error("paths.pretrained differs from the checkpoint used for fine-tuning")
    patch = cfg["train"]["patch"]
    validate_model(ckpt["model"], patch)
    hardware = cfg["device"]
    device = runtime_device(hardware["gpu"], hardware["amp"])
    model, _ = build_hyps_model(ckpt["model"], pretrained)
    model.to(device)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    df = evaluate(model, snapshots, cases, out, "predict", hardware["amp"], patch=patch,
                  min_component=cfg["inference"]["min_component"], with_label=False, save_predictions=True,
                  num_workers=hardware.get("num_workers") or 0)
    volumes = df.rename(columns={"vol_pred_left": "left_cm3", "vol_pred_right": "right_cm3",
                                 "vol_pred_total": "total_cm3"})
    volumes.to_csv(out / "volumes.csv", index=False)
    print(f"segmented {len(df)} images -> {out}")


if __name__ == "__main__":
    main()
