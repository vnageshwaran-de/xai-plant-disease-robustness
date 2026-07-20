"""11_scorecam_fix.py  Does a bounded (min-max) channel weighting restore Score-CAM
stability on the fine-tuned model? Resumable per (img,norm)."""
import os, json, time, numpy as np, torch, torch.nn as nn
import torchvision.models as tvm
import common as C, xai_lib as X

RES = C.RESULTS
OUT = f"{RES}/finetuned/scorecam_fix.json"
FT = "/tmp/ftr/efficientnet_b0_ft.pt"
if not os.path.exists(FT): FT = f"{RES}/finetuned/efficientnet_b0_ft.pt"
NIMG = int(os.environ.get("NIMG", "4"))
sub = json.load(open(f"{RES}/raw/xai_subset.json"))["images"][:NIMG]

ck = torch.load(FT, map_location="cpu"); sd = ck["model"] if "model" in ck else ck
ft = tvm.efficientnet_b0(weights=None); ft.classifier[1] = nn.Linear(ft.classifier[1].in_features, 38)
ft.load_state_dict(sd); ft.eval(); layer = ft.features[-1]

def cam(x, cls, norm):
    acts = {}
    h = layer.register_forward_hook(lambda m,i,o: acts.__setitem__('a', o.detach()))
    with torch.no_grad(): ft(x)
    h.remove()
    up = torch.nn.UpsamplingBilinear2d(size=x.shape[-2:])(acts['a'])
    mn = up.amin((2,3), keepdim=True); mx = up.amax((2,3), keepdim=True)
    upn = (up - mn)/(mx - mn + 1e-8)
    masked = (x[:,None]*upn[:,:,None])[0]
    sc = []
    with torch.no_grad():
        for i in range(0, masked.shape[0], 128): sc.append(ft(masked[i:i+128])[:, cls])
    sc = torch.cat(sc)
    w = torch.softmax(sc, 0) if norm=="softmax" else ((sc-sc.min())/(sc.max()-sc.min()+1e-8))
    if norm=="minmax": w = w/w.sum()
    c = (w[None,:,None,None]*upn).sum(1)[0].numpy(); c=np.maximum(c,0)
    return (c-c.min())/(c.max()-c.min()+1e-8)

res = json.load(open(OUT)) if os.path.exists(OUT) else {}
t0=time.time()
for norm in ["softmax","minmax"]:
    for i,rec in enumerate(sub):
        k=f"{norm}|{i}"
        if k in res: continue
        if time.time()-t0>6 and res: json.dump(res,open(OUT,"w")); print("checkpoint"); raise SystemExit
        _, raw = C.load_tensor(rec["path"]); x=X._norm(raw).unsqueeze(0)
        with torch.no_grad(): cls=int(ft(x).argmax(1))
        c0=cam(x,cls,norm); ps=[]
        for t in X.TRANSFORMS:
            xt,_=X.apply_transform(raw,t); ct=cam(xt,cls,norm)
            al,vm=X.align_cam(c0,t); ps.append(X.pearson(al,ct,vm))
        res[k]=float(np.mean(ps)); json.dump(res,open(OUT,"w"))
json.dump(res,open(OUT,"w"))
for norm in ["softmax","minmax"]:
    vals=[v for k,v in res.items() if k.startswith(norm)]
    if vals: print(f"{norm:8s} mean Pearson stability = {np.mean(vals):.3f}  (n={len(vals)} imgs)")
print("ALL_DONE" if len([k for k in res])>=2*len(sub) else "partial")
