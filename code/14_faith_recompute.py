"""14_faith_recompute.py  Corrected + confidence-normalized deletion/insertion for the
fine-tuned model, clean vs perturbed. Fixes the notebook's insertion bug and removes
the base-rate confound by dividing each AUC by the unmasked probability p_full of that
(possibly perturbed) input. Resumable per (method,image)."""
import os, json, time, numpy as np, torch, torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms.functional as TF
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import common as C, xai_lib as X

RES = C.RESULTS; OUT = f"{RES}/finetuned/faith_norm.jsonl"
FT = "/tmp/ftr/efficientnet_b0_ft.pt"
if not os.path.exists(FT): FT = f"{RES}/finetuned/efficientnet_b0_ft.pt"
CAMS = {"GradCAM":GradCAM,"GradCAM++":GradCAMPlusPlus,"EigenCAM":EigenCAM,"LayerCAM":LayerCAM}
NIMG = int(os.environ.get("NIMG","24")); STEPS=15
sub = json.load(open(f"{RES}/raw/xai_subset.json"))["images"][:NIMG]
ck = torch.load(FT, map_location="cpu"); sd = ck["model"] if "model" in ck else ck
ft = tvm.efficientnet_b0(weights=None); ft.classifier[1]=nn.Linear(ft.classifier[1].in_features,38)
ft.load_state_dict(sd); ft.eval(); layer=[ft.features[-1]]

def prob(t3, cls):
    with torch.no_grad(): return float(torch.softmax(ft(X._norm(t3).unsqueeze(0)),1)[0,cls])
def auc_norm(raw, cam, cls):
    H,W=cam.shape; order=np.argsort(-cam.ravel()); n=H*W; step=max(1,n//STEPS)
    base=TF.gaussian_blur(raw,15,8.0)
    di=raw.clone().reshape(3,-1); ii=base.clone().reshape(3,-1); fr=raw.reshape(3,-1)
    pf=prob(raw,cls)+1e-8
    dp=[prob(raw,cls)]; ip=[prob(base,cls)]
    for s in range(0,n,step):
        idx=order[s:s+step]; di[:,idx]=0; ii[:,idx]=fr[:,idx]
        dp.append(prob(di.reshape(3,H,W),cls)); ip.append(prob(ii.reshape(3,H,W),cls))
    dA=np.trapz(dp,dx=1/(len(dp)-1)); iA=np.trapz(ip,dx=1/(len(ip)-1))
    return dA/pf, iA/pf   # normalized by unmasked confidence

done=set()
if os.path.exists(OUT):
    for l in open(OUT):
        try: r=json.loads(l); done.add((r["method"],r["img"]))
        except: pass
jobs=[(m,i) for m in CAMS for i in range(len(sub)) if (m,i) not in done]
out=open(OUT,"a"); t0=time.time(); n=0
for m,i in jobs:
    if time.time()-t0>36: break
    rec=sub[i]; _,raw=C.load_tensor(rec["path"]); x=X._norm(raw).unsqueeze(0)
    with torch.no_grad(): cls=int(ft(x).argmax(1))
    CAM=CAMS[m]
    with CAM(model=ft,target_layers=layer) as cam: c0=cam(input_tensor=x,targets=[ClassifierOutputTarget(cls)])[0]
    dC,iC=auc_norm(raw,c0,cls)
    dP,iP=[],[]
    for t in X.TRANSFORMS:
        xt,rawt=X.apply_transform(raw,t)
        with CAM(model=ft,target_layers=layer) as cam: ct=cam(input_tensor=xt,targets=[ClassifierOutputTarget(cls)])[0]
        d,ii=auc_norm(rawt,ct,cls); dP.append(d); iP.append(ii)
    out.write(json.dumps({"method":m,"img":i,"del_clean":dC,"ins_clean":iC,
                          "del_pert":float(np.mean(dP)),"ins_pert":float(np.mean(iP))})+"\n"); out.flush(); n+=1
out.close(); print(f"wrote {n}; remaining ~{len(jobs)-n}")
if len(jobs)-n<=0:
    recs=[json.loads(l) for l in open(OUT)]
    print("=== Normalized faithfulness (fine-tuned, n=%d), clean vs perturbed ==="%NIMG)
    for m in CAMS:
        r=[x for x in recs if x["method"]==m]
        print(f"  {m:10s} & {np.mean([x['del_clean'] for x in r]):.3f} & {np.mean([x['del_pert'] for x in r]):.3f} & "
              f"{np.mean([x['ins_clean'] for x in r]):.3f} & {np.mean([x['ins_pert'] for x in r]):.3f} \\\\")
    print("ALL_DONE")
