"""Connected-component filtering, segmentation metrics and hippocampal volumes."""
import numpy as np
from scipy import ndimage


def postprocess(mask, min_size=1000):
    """Remove connected components smaller than `min_size` voxels (1,000 voxels = 1 cm^3 at 1 mm)."""
    m = mask.astype(bool)
    if not m.any():
        return m.astype(np.uint8)
    lab, _ = ndimage.label(m)
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_size
    keep[0] = False
    return keep[lab].astype(np.uint8)


def dice(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    s = a.sum() + b.sum()
    return 100.0 if s == 0 else 200.0 * np.logical_and(a, b).sum() / s


def _surface(m):
    return np.logical_xor(m, ndimage.binary_erosion(m, structure=ndimage.generate_binary_structure(3, 1)))


def hausdorff(gt, pred, spacing):
    """Returns (HD95, HD). HD95 = max(h95(A,B), h95(B,A)), where h95 is the 95th percentile of the distances
    from the surface points of one mask to the nearest surface point of the other."""
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    if not gt.any() or not pred.any():
        return np.nan, np.nan
    both = np.logical_or(gt, pred)
    idx = np.argwhere(both)
    lo = np.maximum(idx.min(0) - 3, 0)
    hi = idx.max(0) + 4
    sl = tuple(slice(l, h) for l, h in zip(lo, hi))
    g, p = gt[sl], pred[sl]
    sg, sp = _surface(g), _surface(p)
    dt_g = ndimage.distance_transform_edt(~sg, sampling=spacing)
    dt_p = ndimage.distance_transform_edt(~sp, sampling=spacing)
    d_pg = dt_g[sp]
    d_gp = dt_p[sg]
    hd95 = max(np.percentile(d_pg, 95), np.percentile(d_gp, 95))
    hd = max(d_pg.max(), d_gp.max())
    return float(hd95), float(hd)


def lr_volumes(mask, affine):
    """Left / right / total volume in cm^3, split by the sign of the world x coordinate (RAS: x < 0 is left)."""
    vox = abs(np.linalg.det(affine[:3, :3])) / 1000.0
    m = mask.astype(bool)
    if not m.any():
        return 0.0, 0.0, 0.0
    ii, jj, kk = np.nonzero(m)
    x = affine[0, 0] * ii + affine[0, 1] * jj + affine[0, 2] * kk + affine[0, 3]
    left = float((x < 0).sum() * vox)
    right = float((x >= 0).sum() * vox)
    return left, right, left + right


def evaluate_case(pred, gt, affine):
    out = {}
    pl, pr, pt = lr_volumes(pred, affine)
    out.update(vol_pred_left=pl, vol_pred_right=pr, vol_pred_total=pt)
    if gt is not None:
        if gt.shape != pred.shape:
            raise ValueError("Prediction and label must have the same shape")
        axes = affine[:3, :3]
        spacing = np.linalg.norm(axes, axis=0)
        directions = axes / spacing
        if not np.allclose(directions.T @ directions, np.eye(3), atol=1e-4):
            raise ValueError("HD95 requires an orthogonal voxel grid")
        gl, gr, gtot = lr_volumes(gt, affine)
        hd95, hd = hausdorff(gt, pred, spacing)
        out.update(dice=dice(pred, gt), hd95=hd95, hd=hd, vol_gt_left=gl, vol_gt_right=gr, vol_gt_total=gtot)
    return out
