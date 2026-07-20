"""16_self_consistency.py  Control for Control 2: is Grad-CAM's high class-sensitivity
genuine class-discrimination, or just a noisy map decorrelating against anything?
Self-inconsistency = 1 - r between CAM(x, cls) and CAM(x + small jitter, cls).
If self-inconsistency << class-sensitivity, the class-sensitivity claim is clean."""
import os, json, time, numpy as np, torch, torch.nn as nn
import torchvision.models as tvm
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import common as C, xai_lib as X

RES = C.RESULTS; OUT = f"{RES}/finetuned/self_consistency.jsonl"
FT = "/tmp/ftr/efficientnet_b0_ft.pt"
if not os.path.exists(FT): FT = f"{RES}/finetuned/efficientnet_b0_ft.pt"
CAMS = {"GradCAM":GradCAM,"GradCAM++":GradCAMPlusPlus,"EigenCAM":EigenCAM,"LayerCAM":LayerCAM}
JIT = 0.02  # minor input jitter (Gaussian sigma on [0,1])
sub = json.load(open(f"{RES}/raw/xai_subset.json"))["images"]
ck = torch.load(FT, map_location="cpu"); sd = ck["model"] if "model" in ck else ck
ft = tvm.efficientnet_b0(weights=None); ft.classifier[1]=nn.Linear(ft.classifier[1].in_features,38)
ft.load_state_dict(sd); ft.eval(); layer=[ft.features[-1]]

done=set()
if os.path.exists(OUT):
    for l in open(OUT):
        try: r=json.loads(l); done.add((r["method"],r["img"]))
        except: pass
jobs=[(m,i) for m in CAMS for i in range(len(sub)) if (m,i) not in done]
out=open(OUT,"a"); t0=time.time(); n=0; g=torch.Generator().manual_seed(123)
for m,i in jobs:
    if time.time()-t0>36: break
    rec=sub[i]; _,raw=C.load_tensor(rec["path"]); x=X._norm(raw).unsqueeze(0)
    with torch.no_grad(): cls=int(ft(x).argmax(1))
    CAM=CAMS[m]; tgt=[ClassifierOutputTarget(cls)]
    with CAM(model=ft,target_layers=layer) as cam: c0=cam(input_tensor=x,targets=tgt)[0]
    rawj=(raw+torch.randn(raw.shape,generator=g)*JIT).clamp(0,1); xj=X._norm(rawj).unsqueeze(0)
    with CAM(model=ft,target_layers=layer) as cam: cj=cam(input_tensor=xj,targets=tgt)[0]
    vm=np.ones(c0.shape,bool)
    out.write(json.dumps({"method":m,"img":i,"self_inconsistency":float(1-max(0.0,X.pearson(c0,cj,vm)))})+"\n"); out.flush(); n+=1
out.close(); print(f"wrote {n}; remaining ~{len(jobs)-n}")
if len(jobs)-n<=0:
    recs=[json.loads(l) for l in open(OUT)]
    ctrl=json.load(open(f"{RES}/finetuned/controls_summary.json"))
    print("method     class_sensitivity  self_inconsistency")
    for m in CAMS:
        si=np.mean([x['self_inconsistency'] for x in recs if x['method']==m])
        cs=ctrl['class_sensitivity'][m]
        print(f"  {m:10s}   {cs:.3f}            {si:.3f}")
    print("ALL_DONE")
