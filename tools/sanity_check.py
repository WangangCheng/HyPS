"""Check output equivalence and trainable parameters before fine-tuning."""
import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hyps.config import load_config, select_gpu  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config, require={"paths": ["pretrained"], "device": ["gpu"], "model": ["arch", "rank", "scale"]})
    select_gpu(cfg)
    import torch
    from hyps.engine import runtime_device
    from hyps.hyps_linear import HYPS_KEYS, apply_hyps, count_params, is_decoder
    from hyps.models import build

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = runtime_device(cfg["device"]["gpu"], "none")
    model_cfg = cfg["model"]
    scope = model_cfg.get("train", "decoder")
    base = build(model_cfg["arch"], pretrained=cfg["paths"]["pretrained"]).eval()
    baseline_count = sum(parameter.numel() for parameter in base.parameters())
    model = copy.deepcopy(base)
    n = apply_hyps(model, model_cfg["arch"], r=model_cfg["rank"], s=model_cfg["scale"],
                   layers=model_cfg.get("layers"), train=scope)
    base.to(device)
    model.to(device)
    patch = cfg["train"].get("patch") or 128
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(1, 1, patch, patch, patch, generator=generator).to(device)
    with torch.no_grad():
        expected, actual = base(x), model(x)
    rel = ((actual - expected).norm() / expected.norm().clamp_min(1e-12)).item()
    print(f"HyPS layers: {n}. Relative output difference at initialization: {rel:.2e} (expected < 1e-4).")
    del base, expected, actual

    model.train()
    model(x).float().mean().backward()
    bad = []
    for name, parameter in model.named_parameters():
        adapter = name.rsplit(".", 1)[-1] in HYPS_KEYS
        decoder = is_decoder(name) if scope == "decoder" else name.startswith("out.")
        if parameter.requires_grad != (adapter or decoder):
            bad.append(name)
    no_grad = [k for k, p in model.named_parameters() if p.requires_grad and p.grad is None]
    print(f"Parameters with incorrect training scope: {len(bad)}. Trainable without gradient: {len(no_grad)}.")
    c = count_params(model)
    print(f"trainable parameters: encoder {c['trainable_encoder'] / 1e6:.2f} M, decoder {c['trainable_decoder'] / 1e6:.2f} M, "
          f"total {c['trainable_total'] / 1e6:.2f} M. Backbone {baseline_count / 1e6:.2f} M.")
    ok = rel < 1e-4 and not bad and not no_grad
    print("OK" if ok else "CHECK FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
