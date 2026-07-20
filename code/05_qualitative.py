"""
05_qualitative.py
Qualitative heatmap montages. Representative images chosen by the stability
distribution: a high-stability (success) case and a low-stability (failure) case.
Rows = XAI methods; columns = original + each transformation (CAM overlays).
"""
import os, json
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, EigenCAM, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import common as C, model_utils as MU, xai_lib as X

RES = C.RESULTS
FIG = RES + "/figures"
CAMS = [("GradCAM", GradCAM), ("GradCAM++", GradCAMPlusPlus), ("ScoreCAM", ScoreCAM),
        ("EigenCAM", EigenCAM), ("LayerCAM", LayerCAM)]
CONDS = ["identity"] + X.TRANSFORMS
BB = "efficientnet_b0"

sub = json.load(open(f"{RES}/raw/xai_subset.json"))["images"]
recs = [json.loads(l) for l in open(f"{RES}/raw/xai_records.jsonl")
        if json.loads(l)["backbone"] == BB]
# per-image mean stability (over fast methods & transforms)
from collections import defaultdict
per_img = defaultdict(list)
for r in recs:
    if r["method"] in ("GradCAM", "GradCAM++", "EigenCAM", "LayerCAM"):
        per_img[r["img_idx"]].append(np.mean([r["stability"][t]["pearson"] for t in X.TRANSFORMS]))
score = {i: np.mean(v) for i, v in per_img.items()}
order = sorted(score, key=score.get)
# also restrict to images that have ScoreCAM (idx<12) so montage rows are complete
have_score = [i for i in order if i < 12]
fail_idx = have_score[0]                       # lowest stability with ScoreCAM
succ_idx = have_score[len(have_score)//2]      # median stability with ScoreCAM
print(f"success idx={succ_idx} (r={score[succ_idx]:.3f}), failure idx={fail_idx} (r={score[fail_idx]:.3f})")

model, _ = MU.load_model(BB)
tl = [MU.target_layer(model, BB)]

def montage(idx, fname, title):
    if os.path.exists(f"{FIG}/{fname}"):
        print("skip existing", fname); return
    rec = sub[idx]; _, raw = C.load_tensor(rec["path"])
    xin = X._norm(raw).unsqueeze(0)
    with torch.no_grad():
        cls = int(model(xin).argmax(1))
    tgt = [ClassifierOutputTarget(cls)]
    nrows, ncols = len(CAMS) + 1, len(CONDS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.0 * ncols, 2.0 * nrows))
    # top row: transformed input images
    for j, cond in enumerate(CONDS):
        _, rawt = X.apply_transform(raw, cond)
        axes[0, j].imshow(rawt.permute(1, 2, 0).numpy())
        axes[0, j].set_title(cond, fontsize=10)
        axes[0, j].axis("off")
    axes[0, 0].set_ylabel("input", fontsize=10)
    for k, (mname, CAM) in enumerate(CAMS):
        for j, cond in enumerate(CONDS):
            xt, rawt = X.apply_transform(raw, cond)
            with CAM(model=model, target_layers=tl) as cam:
                if mname == "ScoreCAM": cam.batch_size = 128
                g = cam(input_tensor=xt, targets=tgt)[0]
            rgb = rawt.permute(1, 2, 0).numpy()
            over = show_cam_on_image(rgb, g, use_rgb=True)
            axes[k + 1, j].imshow(over); axes[k + 1, j].axis("off")
        axes[k + 1, 0].axis("on"); axes[k + 1, 0].set_xticks([]); axes[k + 1, 0].set_yticks([])
        axes[k + 1, 0].set_ylabel(mname, fontsize=10)
    fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(f"{FIG}/{fname}", dpi=140); plt.close()
    print("saved", fname)

montage(succ_idx, "fig5_qualitative_success.png",
        f"Stable explanations (EfficientNet-B0) — {sub[succ_idx]['class']}")
montage(fail_idx, "fig6_qualitative_failure.png",
        f"Unstable explanations (EfficientNet-B0) — {sub[fail_idx]['class']}")
