"""Transformation protocol, explanation-stability metrics, faithfulness metrics."""
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from skimage.metrics import structural_similarity as ssim_fn
import common as C

# ----------------- Transformation protocol -----------------
# All transforms operate on a raw [0,1] CHW tensor; we renormalise for the model.
ROT_DEG = 15
BLUR_SIGMA = 2.0
BRIGHT_FACTOR = 1.3
NOISE_SIGMA = 0.08

def _norm(raw):
    return TF.normalize(raw.clamp(0, 1), C.IMAGENET_MEAN, C.IMAGENET_STD)

def apply_transform(raw, name):
    """Return (normalized_input, raw_transformed). raw is CHW [0,1]."""
    if name == "identity":
        r = raw.clone()
    elif name == "blur":
        r = TF.gaussian_blur(raw, kernel_size=9, sigma=BLUR_SIGMA)
    elif name == "rotate":
        r = TF.rotate(raw, ROT_DEG, interpolation=TF.InterpolationMode.BILINEAR)
    elif name == "brightness":
        r = (raw * BRIGHT_FACTOR).clamp(0, 1)
    elif name == "noise":
        g = torch.randn_like(raw) * NOISE_SIGMA
        r = (raw + g).clamp(0, 1)
    else:
        raise ValueError(name)
    return _norm(r).unsqueeze(0), r

TRANSFORMS = ["blur", "rotate", "brightness", "noise"]

def align_cam(cam_orig, name):
    """Map an original-image CAM into the transformed image's frame.
    Photometric transforms: identity. Rotation: rotate the CAM by ROT_DEG.
    Returns (aligned_cam, valid_mask) both HxW float arrays."""
    H, W = cam_orig.shape
    if name == "rotate":
        t = torch.tensor(cam_orig).unsqueeze(0)
        rc = TF.rotate(t, ROT_DEG, interpolation=TF.InterpolationMode.BILINEAR)[0].numpy()
        ones = torch.ones(1, H, W)
        vm = TF.rotate(ones, ROT_DEG, interpolation=TF.InterpolationMode.NEAREST)[0].numpy() > 0.5
        return rc, vm
    return cam_orig.copy(), np.ones((H, W), dtype=bool)

# ----------------- Explanation-stability metrics -----------------
def pearson(a, b, mask):
    x = a[mask].ravel(); y = b[mask].ravel()
    if x.std() < 1e-8 or y.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])

def ssim_metric(a, b, mask):
    # zero-out invalid region so both maps share support
    aa = a.copy(); bb = b.copy()
    aa[~mask] = 0; bb[~mask] = 0
    return float(ssim_fn(aa, bb, data_range=1.0))

def iou_topk(a, b, mask, k=0.2):
    va = a[mask]; vb = b[mask]
    if va.size == 0:
        return 0.0
    ta = np.quantile(va, 1 - k); tb = np.quantile(vb, 1 - k)
    ba = (a >= ta) & mask; bb = (b >= tb) & mask
    inter = np.logical_and(ba, bb).sum(); union = np.logical_or(ba, bb).sum()
    return float(inter / union) if union > 0 else 0.0

# ----------------- Faithfulness metrics -----------------
def _prob(model, x, cls):
    with torch.no_grad():
        return float(F.softmax(model(x), 1)[0, cls])

def deletion_insertion(model, raw, cam, cls, steps=20):
    """Deletion & insertion AUC. cam: HxW [0,1] at input resolution.
    Deletion: remove most-salient pixels first (replace by 0). Lower AUC = better.
    Insertion: start from blurred baseline, add most-salient pixels. Higher = better."""
    H, W = cam.shape
    order = np.argsort(-cam.ravel())            # descending saliency
    n = H * W
    step = max(1, n // steps)
    x_full = _norm(raw).unsqueeze(0)
    baseline = TF.gaussian_blur(raw, kernel_size=15, sigma=8.0)  # insertion start
    del_probs, ins_probs = [], []
    del_img = raw.clone(); ins_img = baseline.clone()
    del_probs.append(_prob(model, _norm(del_img).unsqueeze(0), cls))
    ins_probs.append(_prob(model, _norm(ins_img).unsqueeze(0), cls))
    flat_raw = raw.reshape(3, -1); flat_base = baseline.reshape(3, -1)
    del_flat = del_img.reshape(3, -1); ins_flat = ins_img.reshape(3, -1)
    for s in range(0, n, step):
        idx = order[s:s + step]
        del_flat[:, idx] = 0.0                  # delete salient -> zero
        ins_flat[:, idx] = flat_raw[:, idx]     # insert salient -> real pixels
        del_probs.append(_prob(model, _norm(del_flat.reshape(3, H, W)).unsqueeze(0), cls))
        ins_probs.append(_prob(model, _norm(ins_flat.reshape(3, H, W)).unsqueeze(0), cls))
    del_auc = float(np.trapz(del_probs, dx=1.0 / (len(del_probs) - 1)))
    ins_auc = float(np.trapz(ins_probs, dx=1.0 / (len(ins_probs) - 1)))
    return del_auc, ins_auc

def average_drop(model, raw, cam, cls):
    """Average Drop (%): use CAM as soft mask; lower = better."""
    p_full = _prob(model, _norm(raw).unsqueeze(0), cls)
    m = torch.tensor(cam).unsqueeze(0)          # 1xHxW in [0,1]
    masked = (raw * m).clamp(0, 1)
    p_mask = _prob(model, _norm(masked).unsqueeze(0), cls)
    return float(max(0.0, (p_full - p_mask)) / (p_full + 1e-8) * 100.0)
