"""Build the segmentation backbones with one input channel and two outputs."""
import torch


def build(arch, pretrained=None, use_checkpoint=False):
    if arch == "swinunetr":
        from monai.networks.nets import SwinUNETR
        model = SwinUNETR(img_size=(128, 128, 128), in_channels=1, out_channels=2, feature_size=48,
                         use_checkpoint=use_checkpoint)
    elif arch == "unetr":
        from monai.networks.nets import UNETR
        model = UNETR(in_channels=1, out_channels=2, img_size=(128, 128, 128), feature_size=16,
                      hidden_size=768, mlp_dim=3072, num_heads=12, proj_type="conv",
                      norm_name="instance", res_block=True, qkv_bias=False)
    else:
        raise ValueError(f"unknown architecture: {arch}")
    if pretrained:
        checkpoint = torch.load(pretrained, map_location="cpu", weights_only=True)
        state = checkpoint.get("state_dict", checkpoint)
        state = {name.removeprefix("module.").removeprefix("_orig_mod."): tensor
                 for name, tensor in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [k for k in missing if not k.endswith("relative_position_index")]
        if missing or unexpected:
            raise RuntimeError(f"pre-trained weights do not match the model: missing={missing[:10]} "
                               f"unexpected={unexpected[:10]}")
    return model
