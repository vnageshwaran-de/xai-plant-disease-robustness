"""
09_finetune.py  END-TO-END fine-tuning of EfficientNet-B0 on the PlantVillage
subset (addresses reviewer #2). Resumable: caches decoded tensors, checkpoints
after each epoch, evaluates clean accuracy/macro-F1. CPU-friendly.
"""
import os, sys, time, json
import numpy as np, torch, torch.nn as nn
import torchvision.models as tvm
from sklearn.metrics import accuracy_score, f1_score
import common as C

torch.set_num_threads(4); torch.manual_seed(42); np.random.seed(42)
RES = C.RESULTS; os.makedirs(RES + "/models", exist_ok=True)
EPOCHS = 5; BS = 16; BUDGET = 28.0
CKPT = f"{RES}/models/efficientnet_b0_ft.pt"
METJ = f"{RES}/tables/ft_efficientnet_b0.json"
def _atomic_save(obj, path):
    tmp = path + ".tmp"; torch.save(obj, tmp); os.replace(tmp, path)

def _cache_split(split):
    """Cache one split's normalized image tensors atomically; resumable."""
    path = f"{C.CACHE}/ft_{split}.pt"
    paths, ys, classes = C.list_split(split)
    if os.path.exists(path):
        try:
            d = torch.load(path); return d["X"], d["Y"], classes
        except Exception:
            os.remove(path)
    X = torch.stack([C.load_tensor(p)[0] for p in paths])
    _atomic_save({"X": X, "Y": torch.tensor(ys)}, path)
    return X, torch.tensor(ys), classes

def preload():
    Xtr, Ytr, classes = _cache_split("train")
    Xte, Yte, _ = _cache_split("test")
    return Xtr, Ytr, Xte, Yte, classes

Xtr, Ytr, Xte, Yte, classes = preload()
N = len(Xtr)

def build():
    m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(classes))
    return m

model = build()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
cur_epoch = 0; cur_step = 0; hist = []
if os.path.exists(CKPT):
    try:
        ck = torch.load(CKPT)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); cur_epoch = ck["epoch"]; cur_step = ck.get("step", 0); hist = ck["hist"]
        print(f"resumed at epoch {cur_epoch} step {cur_step}")
    except Exception as e:
        print(f"corrupt checkpoint ignored ({type(e).__name__}); starting fresh")
        cur_epoch = 0; cur_step = 0; hist = []
if cur_epoch >= EPOCHS:
    print("FT_DONE", json.dumps(hist[-1])); sys.exit(0)

lossf = nn.CrossEntropyLoss()
def save(epoch, step):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "epoch": epoch, "step": step, "hist": hist, "classes": classes}, CKPT)

def evaluate():
    model.eval(); preds = []
    with torch.no_grad():
        for i in range(0, len(Xte), 64):
            preds.append(model(Xte[i:i + 64]).argmax(1))
    preds = torch.cat(preds).numpy()
    return accuracy_score(Yte.numpy(), preds), f1_score(Yte.numpy(), preds, average="macro")

t0 = time.time()
epoch = cur_epoch
while epoch < EPOCHS:
    model.train()
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(42 + epoch))
    steps = list(range(0, N, BS))
    for si in range(cur_step, len(steps)):
        idx = perm[steps[si]:steps[si] + BS]
        xb = Xtr[idx].clone(); yb = Ytr[idx]
        flip = torch.rand(len(idx)) < 0.5
        xb[flip] = torch.flip(xb[flip], dims=[-1])
        opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
        if time.time() - t0 > BUDGET:
            save(epoch, si + 1); print(f"checkpoint epoch {epoch} step {si+1}/{len(steps)}"); sys.exit(0)
    cur_step = 0
    sched.step()
    acc, mf1 = evaluate()
    hist.append({"epoch": epoch + 1, "accuracy": float(acc), "macro_f1": float(mf1)})
    print(f"epoch {epoch+1}/{EPOCHS} acc={acc:.4f} f1={mf1:.4f}")
    epoch += 1
    save(epoch, 0)
    json.dump({"backbone": "efficientnet_b0_ft", "history": hist,
               "final": hist[-1], "n_train": N, "n_test": len(Xte)}, open(METJ, "w"), indent=1)
    if time.time() - t0 > BUDGET and epoch < EPOCHS:
        print(f"checkpoint after epoch {epoch}"); sys.exit(0)
print("FT_DONE", json.dumps(hist[-1]))
