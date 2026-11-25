"""Load subject splits and preprocessed NIfTI volumes."""
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


def load_split(split_file, data_root=None):
    split = json.loads(Path(split_file).read_text(encoding="utf-8"))
    if not isinstance(split, dict):
        raise ValueError("The split must be a JSON object")
    root = Path(data_root) if data_root else Path(split_file).parent
    result, ids, images = {}, set(), set()
    for part in ("train", "val", "test"):
        cases = split.get(part, [])
        if not isinstance(cases, list):
            raise ValueError(f"The {part} split must be a list of cases")
        items = []
        for case in cases:
            image = (root / case["image"]).resolve()
            label = (root / case["label"]).resolve() if case.get("label") else None
            subject = validate_id(case.get("id") or nifti_id(image))
            if subject.casefold() in ids or image in images:
                raise ValueError(f"Duplicate case or image in the split: {subject}")
            ids.add(subject.casefold())
            images.add(image)
            items.append(dict(id=subject, image=image, label=label))
        result[part] = items
    if not result["train"]:
        raise ValueError("The split must contain at least one training case")
    return result


def nifti_id(path):
    return path.name[:-7] if path.name.endswith(".nii.gz") else path.stem


def validate_id(subject):
    if not isinstance(subject, str) or not re.fullmatch(r"[\w][\w.-]*", subject):
        raise ValueError("Case IDs must start with a letter, digit or underscore and contain only letters, "
                         "digits, dots, hyphens or underscores")
    return subject


def nifti_volume(path):
    image = nib.load(path)
    data = np.asarray(image.dataobj)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3 or not all(data.shape):
        raise ValueError(f"Expected a single 3D volume: {path}")
    if not np.isfinite(data).all():
        raise ValueError(f"Volume contains non-finite values: {path}")
    units = image.header.get_xyzt_units()[0]
    affine = image.affine.copy()
    affine[:3] *= {"unknown": 1, "mm": 1, "meter": 1000, "micron": 0.001}[units]
    if not np.isfinite(affine).all() or np.linalg.det(affine[:3, :3]) == 0:
        raise ValueError(f"Invalid NIfTI affine: {path}")
    return data, affine


def zscore_positive(image):
    brain = image > 0
    if brain.any():
        values = image[brain]
        image[brain] = (values - values.mean()) / (values.std() + 1e-8)
    return image


def load_case(case, with_label=True, normalize=True):
    """Return the image, optional binary label and affine in millimetres."""
    image, affine = nifti_volume(case["image"])
    image = image.astype(np.float32)
    if normalize:
        zscore_positive(image)
    label = None
    if with_label and case.get("label"):
        data, label_affine = nifti_volume(case["label"])
        if data.shape != image.shape or not np.allclose(affine, label_affine, atol=1e-4, rtol=0):
            raise ValueError(f"Image and label grids differ for case {case['id']}")
        label = (data > 0.5).astype(np.uint8)
    return image, label, affine


def cache_key(*paths):
    sources = []
    for path in paths:
        source = Path(path).resolve(strict=True)
        stat = source.stat()
        sources.append((str(source), stat.st_size, stat.st_mtime_ns))
    return hashlib.sha256(json.dumps(sources).encode("utf-8")).hexdigest()[:24]


def save_array(path, array):
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as tmp:
        pending = Path(tmp.name)
        try:
            np.save(tmp, array, allow_pickle=False)
            tmp.close()
            os.replace(pending, path)
        except PermissionError:
            if not path.is_file():
                raise
        finally:
            tmp.close()
            pending.unlink(missing_ok=True)


def cache_case(case, cache_dir):
    if not case.get("label"):
        raise ValueError(f"Missing label for case {case['id']}")
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    key = cache_key(case["image"], case["label"])
    paths = tuple(root / f"{key}_{kind}.npy" for kind in ("img", "lab", "aff"))
    if not all(path.is_file() for path in paths):
        for path, array in zip(paths, load_case(case)):
            if not path.is_file():
                save_array(path, array)
    return paths


def load_cached(case, cache_dir, mmap=True):
    fi, fl, fa = cache_case(case, cache_dir)
    mode = "r" if mmap else None
    return np.load(fi, mmap_mode=mode), np.load(fl, mmap_mode=mode), np.load(fa)


def pad_to_patch(array, size):
    padding = [(0, max(0, size - length)) for length in array.shape]
    return np.pad(array, padding) if any(right for _, right in padding) else array


def augment_patch(image, label, intensity):
    for axis in range(3):
        if np.random.rand() < 0.5:
            image = np.flip(image, axis)
            label = np.flip(label, axis)
    image = image * np.random.uniform(1 - intensity, 1 + intensity) + np.random.uniform(-intensity, intensity)
    return (torch.from_numpy(np.ascontiguousarray(image[None], dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(label, dtype=np.int64)))


class TrainSet(Dataset):
    """Sample and augment one cubic patch per training case."""

    def __init__(self, cases, cache_dir, patch=128, intensity=0.1):
        self.cases = list(cases)
        self.cache_dir = cache_dir
        self.patch = patch
        self.intensity = intensity
        for c in self.cases:
            if not c.get("label"):
                raise ValueError(f"training case {c['id']} has no label")
            cache_case(c, cache_dir)

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, i):
        img, lab, _ = load_cached(self.cases[i], self.cache_dir)
        p = self.patch
        starts = [np.random.randint(max(1, length - p + 1)) for length in img.shape]
        crop = tuple(slice(start, start + p) for start in starts)
        return augment_patch(pad_to_patch(img[crop], p), pad_to_patch(lab[crop], p), self.intensity)


class EvalSet(Dataset):
    """Whole-volume inference; also returns the label when available."""

    def __init__(self, cases, cache_dir=None, with_label=True):
        self.cases = list(cases)
        self.cache_dir = cache_dir
        self.with_label = with_label

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, i):
        c = self.cases[i]
        if self.cache_dir and self.with_label and c.get("label"):
            img, lab, aff = load_cached(c, self.cache_dir, mmap=False)
        else:
            img, lab, aff = load_case(c, self.with_label)
        return c["id"], torch.from_numpy(np.ascontiguousarray(img)[None]), lab, aff


def collate_eval(batch):
    assert len(batch) == 1
    return batch[0]
