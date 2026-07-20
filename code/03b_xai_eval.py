"""
03b_xai_eval.py
Explanation-stability (under transforms) + faithfulness per (backbone, method, image).
Resumable: appends one JSON line per completed record to xai_records.jsonl; skips
records already present. Runs within a ~40s time budget then exits (re-run to continue).
"""
import os, json, time, sys
import numpy as np
import torch
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, EigenCAM, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import common as C, model_utils as MU, xai_lib as X

RES = C.RESULTS
RECF = f"{RES}/raw/xai_records.jsonl"
TIME_BUDGET = 40.0
CAMS = {"GradCAM": GradCAM, "GradCAM++": GradCAMPlusPlus, "EigenCAM": EigenCAM,
        "LayerCAM": LayerCAM, "ScoreCAM": ScoreCAM}
FAST = ["GradCAM", "GradCAM++", "EigenCAM", "LayerCAM"]
SCORE_SUBSET = 12  # ScoreCAM only on first 12 images (CPU cost), efficientnet only

sub = json.load(open(f"{RES}/raw/xai_subset.json"))["images"]

def method_list(backbone, img_idx):
    if backbone == "resnet50":
        return FAST
    ms = list(FAST)
    if img_idx < SCORE_SUBSET:
        ms.append("ScoreCAM")
    return ms

# build job order
jobs = []
for backbone in ["resnet50", "efficientnet_b0"]:
    for i, rec in enumerate(sub):
        for m in method_list(backbone, i):
            jobs.append((backbone, m, i))

done = set()
if os.path.exists(RECF):
    for line in open(RECF):
        try:
            r = json.loads(line); done.add((r["backbone"], r["method"], r["img_idx"]))
        except Exception:
            pass
todo = [j for j in jobs if j not in done]
print(f"total jobs={len(jobs)} done={len(done)} todo={len(todo)}")
if not todo:
    print("ALL_DONE"); sys.exit(0)

# group by backbone to avoid reloading model repeatedly
t0 = time.time()
cur_bb = None; model = None; tlayer = None
out = open(RECF, "a")
n_new = 0
for backbone, method, i in todo:
    elapsed = time.time() - t0
    if elapsed > TIME_BUDGET:
        break
    if method == "ScoreCAM" and elapsed > 8:
        continue  # never start a slow ScoreCAM record late; defer to next run
    if backbone != cur_bb:
        model, _ = MU.load_model(backbone)
        tlayer = [MU.target_layer(model, backbone)]
        cur_bb = backbone
    rec = sub[i]
    _, raw = C.load_tensor(rec["path"])
    xin = X._norm(raw).unsqueeze(0)
    with torch.no_grad():
        logits = model(xin); cls = int(logits.argmax(1))
        prob = float(torch.softmax(logits, 1)[0, cls])
    tgt = [ClassifierOutputTarget(cls)]
    CAM = CAMS[method]
    torch.manual_seed(42)
    # original CAM
    with CAM(model=model, target_layers=tlayer) as cam:
        if method == "ScoreCAM":
            cam.batch_size = 128
        cam0 = cam(input_tensor=xin, targets=tgt)[0]
    # faithfulness (on original)
    del_auc, ins_auc = X.deletion_insertion(model, raw, cam0, cls, steps=20)
    avg_drop = X.average_drop(model, raw, cam0, cls)
    # stability under transforms
    stab = {}
    for t in X.TRANSFORMS:
        xt, rawt = X.apply_transform(raw, t)
        with CAM(model=model, target_layers=tlayer) as cam:
            if method == "ScoreCAM":
                cam.batch_size = 128
            camt = cam(input_tensor=xt, targets=tgt)[0]
        aligned, vm = X.align_cam(cam0, t)
        stab[t] = {"pearson": X.pearson(aligned, camt, vm),
                   "ssim": X.ssim_metric(aligned, camt, vm),
                   "iou": X.iou_topk(aligned, camt, vm)}
    r = {"backbone": backbone, "method": method, "img_idx": i,
         "class": rec["class"], "pred": cls, "prob_orig": prob,
         "deletion_auc": del_auc, "insertion_auc": ins_auc, "average_drop": avg_drop,
         "stability": stab}
    out.write(json.dumps(r) + "\n"); out.flush(); n_new += 1
out.close()
print(f"processed {n_new} new records in {time.time()-t0:.0f}s; remaining ~{len(todo)-n_new}")
