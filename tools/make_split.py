"""Create a seeded subject split from paired NIfTI files."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, help="folder searched recursively for image / label pairs")
    ap.add_argument("--img-suffix", default="_img.nii.gz")
    ap.add_argument("--label-suffix", default="_mask.nii.gz")
    ap.add_argument("--n-train", type=int, required=True)
    ap.add_argument("--n-val", type=int, default=0)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.n_train < 1 or a.n_val < 0:
        ap.error("--n-train must be positive and --n-val must be non-negative")
    if not a.img_suffix or not a.label_suffix or a.img_suffix == a.label_suffix:
        ap.error("Image and label suffixes must be nonempty and different")
    root = Path(a.data_dir)
    cases = []
    seen = set()
    for img in sorted(root.rglob(f"*{a.img_suffix}")):
        cid = img.name[: -len(a.img_suffix)]
        lab = img.with_name(cid + a.label_suffix)
        if img.is_file() and lab.is_file():
            if cid.casefold() in seen:
                ap.error(f"More than one pair has the case ID {cid}")
            seen.add(cid.casefold())
            cases.append(dict(id=cid, image=img.relative_to(root).as_posix(), label=lab.relative_to(root).as_posix()))
    if len(cases) < a.n_train + a.n_val + 1:
        raise SystemExit(f"found only {len(cases)} image/label pairs in {root}")
    perm = np.random.RandomState(a.seed).permutation(len(cases))
    sel = [cases[i] for i in perm]
    sp = dict(train=sel[:a.n_train], val=sel[a.n_train:a.n_train + a.n_val], test=sel[a.n_train + a.n_val:])
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sp, indent=2), encoding="utf-8")
    print(f"{len(sp['train'])} train / {len(sp['val'])} val / {len(sp['test'])} test -> {out}")
    print(f"set paths.data_root to {root.resolve()} in the configuration file")


if __name__ == "__main__":
    main()
