"""
12_ft_controls.py  Fine-tuned model, SAME 38 frozen images, fast CAM methods.
Produces (a) paired fine-tuned stability (reviewer #5) and (b) class-sensitivity
control (reviewer #2): CAM(pred) vs CAM(random wrong class) -> class-sensitivity;
correlate with stability across methods. Resumable per (method,img).
"""
import os, json, time, numpy as np, torch, torch.nn as nn
import torchvision.models as tvm
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import common as C, xai_lib as X

RES = C.RESULTS; OUT = f"{RES}/finetuned/ft_controls.jsonl"
FT = "/tmp/ftr/efficientnet_b0_ft.pt"
if not os.path.exists(FT): FT = f"{RES}/finetuned/efficientnet_b0_ft.pt"
CAMS = {"GradCAM":GradCAM,"GradCAM++":GradCAMPlusPlus,"EigenCAM":EigenCAM,"LayerCAM":LayerCAM}
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
print(f"todo {len(jobs)}")
out=open(OUT,"a"); t0=time.time(); rng=np.random.default_rng(42); n=0
for m,i in jobs:
    if time.time()-t0>36: break
    rec=sub[i]; _,raw=C.load_tensor(rec["path"]); x=X._norm(raw).unsqueeze(0)
    with torch.no_grad(): cls=int(ft(x).argmax(1))
    rc=int(rng.integers(0,38));
    while rc==cls: rc=int(rng.integers(0,38))
    CAM=CAMS[m]
    with CAM(model=ft,target_layers=layer) as cam:
        cam0=cam(input_tensor=x,targets=[ClassifierOutputTarget(cls)])[0]
    with CAM(model=ft,target_layers=layer) as cam:
        camR=cam(input_tensor=x,targets=[ClassifierOutputTarget(rc)])[0]
    vmall=np.ones(cam0.shape,bool)
    class_sens=1.0-max(0.0,X.pearson(cam0,camR,vmall))  # 0 if identical (class-agnostic)
    ps=[]
    for t in X.TRANSFORMS:
        xt,_=X.apply_transform(raw,t)
        with CAM(model=ft,target_layers=layer) as cam:
            camt=cam(input_tensor=xt,targets=[ClassifierOutputTarget(cls)])[0]
        al,vm=X.align_cam(cam0,t); ps.append(X.pearson(al,camt,vm))
    out.write(json.dumps({"method":m,"img":i,"stab_pearson":float(np.mean(ps)),
                          "class_sensitivity":float(class_sens)})+"\n"); out.flush(); n+=1
out.close(); print(f"wrote {n}; remaining ~{len(jobs)-n}")
if len(jobs)-n<=0:
    recs=[json.loads(l) for l in open(OUT)]
    print("\n=== Paired fine-tuned stability (same 38 imgs) & class-sensitivity ===")
    ms=["GradCAM","GradCAM++","EigenCAM","LayerCAM"]
    S={}; CS={}
    for m in ms:
        r=[x for x in recs if x["method"]==m]
        S[m]=np.mean([x["stab_pearson"] for x in r]); CS[m]=np.mean([x["class_sensitivity"] for x in r])
        print(f"  {m:10s} stability={S[m]:.3f}  class_sensitivity={CS[m]:.3f}")
    xs=[CS[m] for m in ms]; ys=[S[m] for m in ms]
    print(f"  corr(class_sensitivity, stability) across methods = {np.corrcoef(xs,ys)[0,1]:.3f}")
    json.dump({"stability":S,"class_sensitivity":CS,
               "corr":float(np.corrcoef(xs,ys)[0,1])}, open(f"{RES}/finetuned/controls_summary.json","w"),indent=1)
    print("ALL_DONE")
