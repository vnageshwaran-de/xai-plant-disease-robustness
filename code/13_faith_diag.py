"""13_faith_diag.py  Diagnose deletion/insertion on the fine-tuned model.
Print curve start/end/AUC (raw) and confidence-normalized AUC, clean vs blurred."""
import os, json, numpy as np, torch, torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms.functional as TF
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import common as C, xai_lib as X

RES = C.RESULTS
FT = "/tmp/ftr/efficientnet_b0_ft.pt"
if not os.path.exists(FT): FT = f"{RES}/finetuned/efficientnet_b0_ft.pt"
ck = torch.load(FT, map_location="cpu"); sd = ck["model"] if "model" in ck else ck
ft = tvm.efficientnet_b0(weights=None); ft.classifier[1]=nn.Linear(ft.classifier[1].in_features,38)
ft.load_state_dict(sd); ft.eval(); layer=[ft.features[-1]]
sub = json.load(open(f"{RES}/raw/xai_subset.json"))["images"][:3]

def curves(raw, cam, cls, steps=20):
    H,W=cam.shape; order=np.argsort(-cam.ravel()); n=H*W; step=max(1,n//steps)
    base=TF.gaussian_blur(raw,15,8.0)
    di=raw.clone().reshape(3,-1); ii=base.clone().reshape(3,-1); fr=raw.reshape(3,-1)
    def p(t):
        with torch.no_grad(): return float(torch.softmax(ft(X._norm(t.reshape(3,H,W)).unsqueeze(0)),1)[0,cls])
    dp=[p(raw.reshape(3,-1))]; ip=[p(base.reshape(3,-1))]
    for s in range(0,n,step):
        idx=order[s:s+step]; di[:,idx]=0; ii[:,idx]=fr[:,idx]
        dp.append(p(di)); ip.append(p(ii))
    return np.array(dp), np.array(ip)

for rec in sub:
    _,raw=C.load_tensor(rec["path"]); x=X._norm(raw).unsqueeze(0)
    with torch.no_grad():
        cls=int(ft(x).argmax(1)); pfull=float(torch.softmax(ft(x),1)[0,cls])
    with GradCAM(model=ft,target_layers=layer) as cam:
        c0=cam(input_tensor=x,targets=[ClassifierOutputTarget(cls)])[0]
    dp,ip=curves(raw,c0,cls)
    dauc=np.trapz(dp,dx=1/(len(dp)-1)); iauc=np.trapz(ip,dx=1/(len(ip)-1))
    print(f"{rec['class'][:22]:22s} p_full={pfull:.3f}")
    print(f"  DELETION  start={dp[0]:.3f} end={dp[-1]:.3f} AUC={dauc:.3f}  norm(/p_full)={dauc/pfull:.3f}")
    print(f"  INSERTION start={ip[0]:.3f} end={ip[-1]:.3f} AUC={iauc:.3f}  norm(/p_full)={iauc/pfull:.3f}")
