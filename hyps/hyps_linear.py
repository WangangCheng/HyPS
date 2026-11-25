"""Principal-subspace adaptation and SeqLoRA for linear layers (Eq. 4)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

HYPS_KEYS = ("pissa_A", "pissa_B", "seq_down", "seq_up")
ENCODER_PREFIX = {"swinunetr": "swinViT.", "unetr": "vit."}


class HyPSLinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, s=1.0):
        super().__init__()
        if type(r) is not int or r < 1:
            raise ValueError("HyPS rank must be a positive integer")
        weight = base.weight.detach()
        W = weight if weight.dtype == torch.float64 else weight.float()
        m, n = W.shape
        self.in_features, self.out_features = n, m
        self.r, self.s = r, s
        self.register_buffer("bias", None if base.bias is None else base.bias.detach().clone())
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        sq = S[:r].sqrt()
        A = U[:, :r] * sq[None, :]
        B = sq[:, None] * Vh[:r, :]
        self.register_buffer("weight", (W - A @ B).to(weight.dtype))
        self.pissa_A = nn.Parameter(A.to(weight.dtype).contiguous())
        self.pissa_B = nn.Parameter(B.to(weight.dtype).contiguous())
        self.seq_down = nn.Parameter(weight.new_empty((r, m)))
        self.seq_up = nn.Parameter(weight.new_zeros((m, r)))
        nn.init.kaiming_uniform_(self.seq_down, a=math.sqrt(5))

    def forward(self, x):
        h = F.linear(x, self.weight + self.pissa_A @ self.pissa_B, self.bias)
        return h + self.s * F.linear(F.relu(F.linear(h, self.seq_down)), self.seq_up)

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, r={self.r}, s={self.s}"


def target_linears(model, arch, layers=None):
    """Find the encoder projections, optionally within selected Swin stages."""
    prefix = ENCODER_PREFIX[arch]
    return [name for name, module in model.named_modules()
            if isinstance(module, nn.Linear) and name.startswith(prefix)
            and (not layers or any(stage in name.split(".") for stage in layers))]


def is_decoder(name):
    return name.startswith("decoder") or name.startswith("out.")


def set_trainable(model, train="decoder"):
    if train not in ("decoder", "head"):
        raise ValueError("HyPS training scope must be 'decoder' or 'head'")
    for name, parameter in model.named_parameters():
        adapter = name.rsplit(".", 1)[-1] in HYPS_KEYS
        decoder = is_decoder(name) if train == "decoder" else name.startswith("out.")
        parameter.requires_grad_(adapter or decoder)


def apply_hyps(model, arch, r=8, s=1.0, layers=None, train="decoder"):
    """Wrap encoder projections and return the number of adapted layers."""
    names = target_linears(model, arch, layers)
    if not names:
        raise ValueError(f"No encoder linear layers found for {arch} with stages {layers}")
    for name in names:
        path, attr = name.rsplit(".", 1)
        parent = model.get_submodule(path)
        setattr(parent, attr, HyPSLinear(getattr(parent, attr), r=r, s=s))
    set_trainable(model, train)
    return len(names)


def count_params(model):
    """Count trainable parameters and include frozen linear weights in the total."""
    enc = dec = tot_tr = tot = 0
    for name, p in model.named_parameters():
        n = p.numel()
        tot += n
        if p.requires_grad:
            tot_tr += n
            if is_decoder(name):
                dec += n
            else:
                enc += n
    for module in model.modules():
        if isinstance(module, HyPSLinear):
            tot += module.weight.numel()
            if module.bias is not None:
                tot += module.bias.numel()
    return dict(trainable_encoder=enc, trainable_decoder=dec, trainable_total=tot_tr, total=tot)


def trainable_state(model):
    return {name: parameter.detach().to(device="cpu", copy=True)
            for name, parameter in model.named_parameters() if parameter.requires_grad}
