"""HyPS for 3D convolutions and its nnU-Net v2 trainer."""
import json
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hyps_linear import HYPS_KEYS


class HyPSConv(nn.Module):
    def __init__(self, base: nn.Conv3d, r=8, s=1.0):
        super().__init__()
        if base.groups != 1:
            raise ValueError("HyPSConv requires an ungrouped convolution")
        if type(r) is not int or r < 1:
            raise ValueError("HyPS rank must be a positive integer")
        self.base = base.requires_grad_(False)
        self.s = s
        cin, cout = base.in_channels, base.out_channels
        self.r = min(r, cout, cin * math.prod(base.kernel_size))
        seq_rank = min(r, cout)
        factory = dict(bias=False, dtype=base.weight.dtype, device=base.weight.device)
        self.pissa_B = nn.Conv3d(cin, self.r, base.kernel_size, stride=base.stride, padding=base.padding,
                                 dilation=base.dilation, padding_mode=base.padding_mode, **factory)
        self.pissa_A = nn.Conv3d(self.r, cout, 1, **factory)
        self.init_pissa()
        self.seq_down = nn.Conv3d(cout, seq_rank, 1, **factory)
        self.seq_up = nn.Conv3d(seq_rank, cout, 1, **factory)
        nn.init.kaiming_uniform_(self.seq_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.seq_up.weight)

    @torch.no_grad()
    def init_pissa(self):
        """Split the loaded kernel into principal factors and a frozen residual."""
        kernel = self.base.weight
        cout = kernel.shape[0]
        dtype = torch.float64 if kernel.dtype == torch.float64 else torch.float32
        matrix = kernel.reshape(cout, -1).to(device="cpu", dtype=dtype)
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
        rank = self.r
        root = S[:rank].sqrt()
        A = U[:, :rank] * root[None]
        B = root[:, None] * Vh[:rank]
        self.pissa_A.weight.copy_(A.reshape(cout, rank, 1, 1, 1))
        self.pissa_B.weight.copy_(B.reshape(rank, *kernel.shape[1:]))
        kernel.copy_((matrix - A @ B).reshape_as(kernel))

    def forward(self, x):
        h = self.base(x) + self.pissa_A(self.pissa_B(x))
        return h + self.s * self.seq_up(F.relu(self.seq_down(h)))


def wrap_encoder(network, r=8, s=1.0):
    """Wrap the encoder's convolutions, retaining references to shared modules."""
    wrapped = {id(layer): HyPSConv(layer, r=r, s=s)
               for layer in network.encoder.modules() if isinstance(layer, nn.Conv3d)}
    for name, layer in list(network.named_modules(remove_duplicate=False)):
        if id(layer) in wrapped:
            path, _, attr = name.rpartition(".")
            setattr(network.get_submodule(path), attr, wrapped[id(layer)])
    return network


def set_trainable(network):
    """Train the encoder adapters and the entire decoder."""
    encoder_ids = {id(parameter) for parameter in network.encoder.parameters()}
    stats = dict(trainable_encoder=0, trainable_decoder=0, trainable_total=0, total=0)
    for name, parameter in network.named_parameters():
        encoder = id(parameter) in encoder_ids
        adapter = any(tag in name.split(".") for tag in HYPS_KEYS)
        parameter.requires_grad_(adapter or not encoder)
        stats["total"] += parameter.numel()
        if parameter.requires_grad:
            stats["trainable_total"] += parameter.numel()
            stats["trainable_encoder" if encoder else "trainable_decoder"] += parameter.numel()
    return stats


@torch.no_grad()
def load_pretrained_into(network, checkpoint_path):
    """Transfer a trusted native nnU-Net checkpoint and split its encoder kernels."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = {key.removeprefix("module.").removeprefix("_orig_mod."): value
              for key, value in checkpoint["network_weights"].items()}
    state = network.state_dict()
    loaded, missing = 0, []
    for key, tensor in state.items():
        parts = key.split(".")
        if "seg_layers" in parts or any(tag in parts for tag in HYPS_KEYS):
            continue
        source_key = key.replace(".base.", ".")
        if source_key in source and source[source_key].shape == tensor.shape:
            state[key] = source[source_key]
            loaded += 1
        else:
            missing.append(key)
    if missing:
        raise RuntimeError(f"Incompatible source checkpoint: {len(missing)} missing or mismatched keys: {missing[:5]}")
    network.load_state_dict(state)
    for layer in network.modules():
        if isinstance(layer, HyPSConv):
            layer.init_pissa()
    return loaded


try:
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
except ModuleNotFoundError as error:
    if error.name != "nnunetv2":
        raise
    nnUNetTrainer = None


def _rank():
    return int(os.environ.get("HYPS_NNUNET_RANK", 8))


if nnUNetTrainer is not None:
    class nnUNetTrainerHyPS(nnUNetTrainer):
        """Use nnU-Net's loss and schedule with the HyPS-Conv trainable parameters."""

        @classmethod
        def build_network_architecture(cls, architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
                                       num_input_channels, num_output_channels, enable_deep_supervision=True):
            network = nnUNetTrainer.build_network_architecture(
                architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
                num_input_channels, num_output_channels, enable_deep_supervision,
            )
            wrap_encoder(network, r=_rank())
            set_trainable(network)
            return network

        def initialize(self):
            self.num_epochs = int(os.environ.get("HYPS_NNUNET_EPOCHS", 100))
            if self.num_epochs < 1:
                raise ValueError("HYPS_NNUNET_EPOCHS must be positive")
            checkpoint = os.environ.get("HYPS_NNUNET_PRETRAINED")
            if not checkpoint or not Path(checkpoint).is_file():
                raise FileNotFoundError("Set HYPS_NNUNET_PRETRAINED to the source nnU-Net checkpoint")
            super().initialize()
            network = self.network.module if self.is_ddp else self.network
            network = getattr(network, "_orig_mod", network)
            loaded = load_pretrained_into(network, checkpoint)
            stats = set_trainable(network)
            stats.update(rank=_rank(), n_pretrained_tensors=loaded)
            self.print_to_log_file(f"HyPS-Conv: {json.dumps(stats)}")

        def configure_optimizers(self):
            params = [parameter for parameter in self.network.parameters() if parameter.requires_grad]
            optimizer = torch.optim.SGD(params, self.initial_lr, weight_decay=self.weight_decay,
                                        momentum=0.99, nesterov=True)
            return optimizer, PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)


def install_trainer():
    """Make the trainer discoverable in the active nnU-Net installation."""
    import nnunetv2

    folder = Path(nnunetv2.__path__[0]) / "training" / "nnUNetTrainer" / "variants" / "hyps"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "__init__.py").touch(exist_ok=True)
    (folder / "hyps_trainer.py").write_text("from hyps.hyps_conv import nnUNetTrainerHyPS\n", encoding="utf-8")
    return folder
