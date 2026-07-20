"""08_iou_k.py  Top-k IoU sensitivity check (k in {0.1,0.2,0.3}), EfficientNet-B0,
fast CAM methods. Reports whether the method ordering by stability is preserved."""
import json, numpy as np, torch
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import common as C, model_utils as MU, xai_lib as X

RES = C.RESULTS; BB = "efficientnet_b0"
METHODS = [("GradCAM", GradCAM), ("GradCAM++", GradCAMPlusPlus),
           ("EigenCAM", EigenCAM), ("LayerCAM", LayerCAM)]
KS = [0.1, 0.2, 0.3]
sub = json.load(open(f"{RES}/raw/xai_subset.json"))["images"]
model, _ = MU.load_model(BB); tl = [MU.target_layer(model, BB)]

agg = {m: {k: [] for k in KS} for m, _ in METHODS}
for rec in sub:
    _, raw = C.load_tensor(rec["path"]); xin = X._norm(raw).unsqueeze(0)
    with torch.no_grad():
        cls = int(model(xin).argmax(1))
    tgt = [ClassifierOutputTarget(cls)]
    for mname, CAM in METHODS:
        with CAM(model=model, target_layers=tl) as cam:
            cam0 = cam(input_tensor=xin, targets=tgt)[0]
        for t in X.TRANSFORMS:
            xt, _ = X.apply_transform(raw, t)
            with CAM(model=model, target_layers=tl) as cam:
                camt = cam(input_tensor=xt, targets=tgt)[0]
            aligned, vm = X.align_cam(cam0, t)
            for k in KS:
                agg[mname][k].append(X.iou_topk(aligned, camt, vm, k=k))

print("Mean top-k IoU (EfficientNet-B0, averaged over 4 transforms x 38 images):")
table = {}
for m, _ in METHODS:
    table[m] = {k: float(np.mean(agg[m][k])) for k in KS}
    print(f"  {m:10s} " + "  ".join(f"k={k}:{table[m][k]:.3f}" for k in KS))
# ordering preserved?
order = {k: [m for m, _ in sorted(METHODS, key=lambda x: -table[x[0]][k])] for k in KS}
print("Ranking by IoU at each k:")
for k in KS: print(f"  k={k}: {order[k]}")
json.dump(table, open(f"{RES}/tables/iou_k_sensitivity.json", "w"), indent=1)
print("saved iou_k_sensitivity.json")
