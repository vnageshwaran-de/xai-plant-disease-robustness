"""15_compact_montage.py  Compact 3-method qualitative montage (4 rows) for the paper."""
import json, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, EigenCAM, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import common as C, model_utils as MU, xai_lib as X

RES = C.RESULTS; FIG = RES + "/figures"; BB = "efficientnet_b0"
CAMS = [("Grad-CAM", GradCAM), ("Eigen-CAM", EigenCAM), ("Layer-CAM", LayerCAM)]
CONDS = ["identity"] + X.TRANSFORMS
sub = json.load(open(f"{RES}/raw/xai_subset.json"))["images"]
idx = 8  # corn common rust (diseased, low stability)
model, _ = MU.load_model(BB); tl = [MU.target_layer(model, BB)]
rec = sub[idx]; _, raw = C.load_tensor(rec["path"]); xin = X._norm(raw).unsqueeze(0)
with torch.no_grad(): cls = int(model(xin).argmax(1))
tgt = [ClassifierOutputTarget(cls)]
fig, axes = plt.subplots(len(CAMS)+1, len(CONDS), figsize=(2.0*len(CONDS), 2.0*(len(CAMS)+1)))
for j, cond in enumerate(CONDS):
    _, rawt = X.apply_transform(raw, cond)
    axes[0, j].imshow(rawt.permute(1,2,0).numpy()); axes[0, j].set_title(cond, fontsize=11); axes[0, j].axis("off")
for k,(mname,CAM) in enumerate(CAMS):
    for j,cond in enumerate(CONDS):
        xt, rawt = X.apply_transform(raw, cond)
        with CAM(model=model, target_layers=tl) as cam:
            g = cam(input_tensor=xt, targets=tgt)[0]
        axes[k+1,j].imshow(show_cam_on_image(rawt.permute(1,2,0).numpy(), g, use_rgb=True)); axes[k+1,j].axis("off")
    axes[k+1,0].axis("on"); axes[k+1,0].set_xticks([]); axes[k+1,0].set_yticks([]); axes[k+1,0].set_ylabel(mname, fontsize=11)
plt.tight_layout()
plt.savefig(f"{FIG}/fig_qual_compact.png", dpi=140, bbox_inches="tight"); print("saved fig_qual_compact.png")
