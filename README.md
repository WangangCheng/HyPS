# HyPS

HyPS combines a principal-subspace update with SeqLoRA to fine-tune pre-trained 3D segmentation networks. This repository provides hippocampus segmentation with SwinUNETR and UNETR, plus HyPS-Conv for nnU-Net v2.

## Quick Start

### 1. Install

Use Python 3.10 or 3.11. Install the PyTorch build for your GPU before installing the other dependencies. [PyTorch installation](https://pytorch.org/get-started/locally/)

```bash
git clone https://github.com/WangangCheng/HyPS.git
cd HyPS
python -m pip install -r requirements.txt
```

### 2. Prepare data and a checkpoint

Use skull-stripped, single-channel T1 scans registered to MNI152 at 1 mm isotropic resolution. Pair each scan with a binary hippocampus mask on the same voxel grid, using names such as `case001_img.nii.gz` and `case001_mask.nii.gz`.

```bash
python tools/make_split.py --data-dir /path/to/hippocampus --n-train 10 --n-val 0 --seed 1 --out splits/split1.json
```

Use `--n-val 10` for an EADC-style split. Repeat with seeds 2 and 3 for additional splits.

Supply a compatible BraTS 2021 checkpoint for your backbone. To create one with the BraTS 2021 T1ce images and masks, run:

```bash
python pretrain_brats.py --brats-dir /path/to/BraTS2021 --out-dir checkpoints/brats \
  --cache-dir cache/brats --batch 2 --accum 2 --workers 4
```

The command writes `checkpoints/brats/model_final.pth`. Add `--arch unetr` to pre-train UNETR.

### 3. Fine-tune and predict

Copy `configs/hyps_swinunetr.yaml` to `configs/local.yaml`. Set `paths.data_root` to the dataset folder, `paths.split_file` to the split JSON and `paths.pretrained` to the backbone checkpoint. For UNETR, set `model.arch: unetr`. [Configuration template](configs/hyps_swinunetr.yaml)

```bash
python tools/sanity_check.py --config configs/local.yaml
python train.py --config configs/local.yaml
python predict.py --config configs/local.yaml --weights outputs/hyps/hyps_weights.pt \
  --input /path/to/preprocessed_scan.nii.gz --output predictions
```

Training writes the HyPS snapshots, the training log and per-case metrics to `paths.output_dir`. Use the same source checkpoint for training and prediction. Prediction accepts a NIfTI file or a folder of preprocessed scans and writes masks to `predictions/pred_predict/` and volumes in cm³ to `predictions/volumes.csv`.

For a short CPU run, set `device.gpu: cpu`, `device.amp: none`, `device.micro_batch: 1` and `train.epochs: 1`. SwinUNETR can use `train.patch: 64` for that check. The paper configuration uses patches of size 128.

Keep local configurations, subject splits, scans, checkpoints and per-subject results outside version control.

## HyPS-Conv

Prepare a source checkpoint and preprocess the target dataset with the source plans. [nnU-Net pre-training and fine-tuning guide](https://github.com/MIC-DKFZ/nnUNet/blob/v2.5.1/documentation/pretraining_and_finetuning.md)

```bash
python -m pip install -r requirements.txt "nnunetv2==2.5.1"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HYPS_NNUNET_PRETRAINED=/path/to/source/checkpoint_final.pth
python -c "from hyps.hyps_conv import install_trainer; install_trainer()"
nnUNetv2_train TARGET_ID 3d_fullres all -tr nnUNetTrainerHyPS -p TRANSFERRED_PLANS
nnUNetv2_predict -i /path/to/images -o predictions/nnunet -d TARGET_ID \
  -c 3d_fullres -f all -tr nnUNetTrainerHyPS -p TRANSFERRED_PLANS --disable_tta
```

Replace `TARGET_ID` and `TRANSFERRED_PLANS` with the target dataset ID and the transferred plans identifier. The trainer defaults to rank 8 and 100 epochs. Set `HYPS_NNUNET_RANK` at both training and prediction if you change the rank.

## Citation

[A Novel Hybrid Parameter-Efficient Fine-Tuning Approach for Hippocampus Segmentation and Alzheimer's Disease Diagnosis](https://arxiv.org/abs/2409.00884)

```bibtex
@article{cheng2024hyps,
  title   = {A Novel Hybrid Parameter-Efficient Fine-Tuning Approach for Hippocampus Segmentation and Alzheimer's Disease Diagnosis},
  author  = {Cheng, Wangang and He, Guanghua and Hu, Keli and Fang, Mingyu and Dong, Liang and Li, Zhong and Zhu, Hancan},
  year    = {2024},
  journal = {arXiv preprint arXiv:2409.00884},
  doi     = {10.48550/arXiv.2409.00884},
  url     = {https://arxiv.org/abs/2409.00884}
}
```

The backbones are provided by [MONAI](https://github.com/Project-MONAI/MONAI) and [nnU-Net](https://github.com/MIC-DKFZ/nnUNet).
