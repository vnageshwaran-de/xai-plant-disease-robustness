"""
03a_pred_robustness.py
(1) Build the explanation-evaluation subset (1 correctly-classified image/class,
    shared across CNN backbones).
(2) Measure prediction-level robustness (prediction consistency + macro-F1 / accuracy
    drop) under each transformation, over a fixed 5-images/class evaluation set.
Resumable via per-(backbone) JSON output.
"""
import os, json, time, random
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score
import common as C, model_utils as MU, xai_lib as X

RES = C.RESULTS
os.makedirs(RES + "/raw", exist_ok=True)
CNN = ["resnet50", "efficientnet_b0"]
SEED = 42

paths, labels, classes = C.list_split("test")
by_cls = {}
for p, y in zip(paths, labels):
    by_cls.setdefault(int(y), []).append(p)

# cached-feature predictions for subset selection
def feat_preds(name):
    D = {"resnet50": 2048, "efficientnet_b0": 1280}[name]
    fte = np.memmap(f"{C.CACHE}/{name}_test_feat.dat", dtype="float32", mode="r",
                    shape=(len(paths), D))
    ck = torch.load(f"{RES}/models/{name}_head.pt", map_location="cpu")
    import torch.nn as nn
    head = nn.Linear(D, len(classes)); head.load_state_dict(ck["head_state"]); head.eval()
    with torch.no_grad():
        pr = head(torch.tensor(np.array(fte))).argmax(1).numpy()
    return pr
pr_res = feat_preds("resnet50"); pr_eff = feat_preds("efficientnet_b0")
correct_both = {}
for i, (p, y) in enumerate(zip(paths, labels)):
    correct_both.setdefault(int(y), [])
    correct_both[int(y)].append((p, pr_res[i] == y and pr_eff[i] == y, pr_eff[i] == y))

# ---- (1) explanation subset: 1 per class, prefer correct-by-both ----
if not os.path.exists(f"{RES}/raw/xai_subset.json"):
    sub = []
    for c in sorted(by_cls):
        cand = correct_both[c]
        pick = next((p for p, both, eff in cand if both), None) \
            or next((p for p, both, eff in cand if eff), None) or cand[0][0]
        sub.append({"path": pick, "label": c, "class": classes[c]})
    json.dump({"seed": SEED, "n_per_class": 1, "images": sub},
              open(f"{RES}/raw/xai_subset.json", "w"), indent=1)
    print("built xai_subset:", len(sub))

# ---- (2) prediction robustness over 5/class eval set ----
rng = random.Random(SEED)
eval_set = []
for c in sorted(by_cls):
    ims = sorted(by_cls[c]); rng.shuffle(ims)
    for p in ims[:5]:
        eval_set.append((p, c))
print("pred-robustness eval set:", len(eval_set))

conds = ["identity"] + X.TRANSFORMS
for name in CNN:
    outp = f"{RES}/raw/predrob_{name}.json"
    if os.path.exists(outp):
        print(name, "already done"); continue
    model, _ = MU.load_model(name)
    torch.manual_seed(SEED)
    preds = {c: [] for c in conds}; ys = []
    t0 = time.time()
    for p, y in eval_set:
        _, raw = C.load_tensor(p); ys.append(y)
        for cond in conds:
            xin, _ = X.apply_transform(raw, cond)
            with torch.no_grad():
                preds[cond].append(int(model(xin).argmax(1)))
    ys = np.array(ys)
    res = {"backbone": name, "n": len(ys), "conditions": {}}
    base_acc = accuracy_score(ys, preds["identity"])
    base_f1 = f1_score(ys, preds["identity"], average="macro")
    for cond in conds:
        pc = float(np.mean(np.array(preds[cond]) == np.array(preds["identity"])))
        acc = accuracy_score(ys, preds[cond]); mf1 = f1_score(ys, preds[cond], average="macro")
        res["conditions"][cond] = {"prediction_consistency": pc, "accuracy": float(acc),
                                   "macro_f1": float(mf1),
                                   "accuracy_drop": float(base_acc - acc),
                                   "macro_f1_drop": float(base_f1 - mf1)}
    json.dump(res, open(outp, "w"), indent=1)
    print(f"{name}: done in {time.time()-t0:.0f}s  base_acc={base_acc:.3f}")
