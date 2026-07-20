"""
02_train.py <backbone>
Extract frozen-backbone features (resumable, memmap-cached), train a linear
classification head, and evaluate on the held-out leaf-split test set.
Outputs: cached features, head weights, and a metrics JSON.
"""
import os, sys, json, time
import numpy as np
import torch, torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
import common as C

name = sys.argv[1]
BATCH = 16
RES_DIR = C.RESULTS
os.makedirs(RES_DIR + "/tables", exist_ok=True)
os.makedirs(RES_DIR + "/models", exist_ok=True)

feats, D, _, kind = C.build_backbone(name)
feats.eval()
for p in feats.parameters():
    p.requires_grad_(False)

def extract(split):
    paths, labels, classes = C.list_split(split)
    N = len(paths)
    fpath = f"{C.CACHE}/{name}_{split}_feat.dat"
    progp = f"{C.CACHE}/{name}_{split}.prog"
    mm = np.memmap(fpath, dtype="float32", mode=("r+" if os.path.exists(fpath) else "w+"),
                   shape=(N, D))
    done = 0
    if os.path.exists(progp):
        done = json.load(open(progp))["done"]
    t0 = time.time()
    i = done
    while i < N:
        batch = paths[i:i+BATCH]
        xs = torch.stack([C.load_tensor(p)[0] for p in batch])
        if kind == "vit":
            xs = torch.nn.functional.interpolate(xs, size=224, mode="bilinear", align_corners=False)
        with torch.no_grad():
            z = feats(xs)
            if kind != "vit":
                z = torch.nn.functional.adaptive_avg_pool2d(z, 1).flatten(1)
        mm[i:i+len(batch)] = z.numpy().astype("float32")
        i += len(batch)
        if time.time() - t0 > 38:   # checkpoint before timeout
            mm.flush(); json.dump({"done": i}, open(progp, "w"))
            print(f"{split}: checkpoint {i}/{N}"); return None, labels, classes
    mm.flush(); json.dump({"done": N}, open(progp, "w"))
    print(f"{split}: features complete {N}/{N} ({time.time()-t0:.0f}s)")
    return np.array(mm), labels, classes

ftr, ytr, classes = extract("train")
if ftr is None: sys.exit(0)   # resume needed
fte, yte, _ = extract("test")
if fte is None: sys.exit(0)

# Train linear head in torch
Xtr = torch.tensor(ftr); Ytr = torch.tensor(ytr)
Xte = torch.tensor(fte); Yte = torch.tensor(yte)
head = nn.Linear(D, len(classes))
opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
lossf = nn.CrossEntropyLoss()
head.train()
for epoch in range(300):
    opt.zero_grad()
    out = head(Xtr)
    loss = lossf(out, Ytr)
    loss.backward(); opt.step()
head.eval()
with torch.no_grad():
    pred = head(Xte).argmax(1).numpy()
acc = accuracy_score(yte, pred)
mf1 = f1_score(yte, pred, average="macro")
metrics = {"backbone": name, "n_train": len(ytr), "n_test": len(yte),
           "n_classes": len(classes), "accuracy": float(acc), "macro_f1": float(mf1),
           "img_size": C.IMG, "feat_dim": D}
json.dump(metrics, open(f"{RES_DIR}/tables/clf_{name}.json", "w"), indent=1)
torch.save({"head_state": head.state_dict(), "classes": classes, "D": D},
           f"{RES_DIR}/models/{name}_head.pt")
print("METRICS", json.dumps(metrics))
