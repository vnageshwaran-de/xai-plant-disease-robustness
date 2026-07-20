"""
06_review_stats.py  (no new model runs)
Addresses reviewer points from the saved per-image records:
 (a) PER-TRANSFORM Holm-corrected pairwise tests (instead of pooling over transforms).
 (b) CI-augmented values for Tables 3-5 (bootstrap 95% CI already stored / recomputed).
Outputs LaTeX-ready snippets + a summary to results/tables/.
"""
import os, json, itertools
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import common as C

RES = C.RESULTS
rng = np.random.default_rng(42)
TR = ["blur", "rotate", "brightness", "noise"]
FAST = ["GradCAM", "GradCAM++", "EigenCAM", "LayerCAM"]

recs = [json.loads(l) for l in open(f"{RES}/raw/xai_records.jsonl")]
df = pd.DataFrame(recs)

def holm(pairs):
    order = np.argsort([p[-1] for p in pairs]); m = len(pairs); adj=[None]*m; run=0
    for rank, idx in enumerate(order):
        run = max(run, (m-rank)*pairs[idx][-1]); adj[idx]=min(1.0, run)
    return adj

# ---------- (a) per-transform pairwise Holm on Pearson ----------
rows=[]
for bb in ["resnet50","efficientnet_b0"]:
    ms=[m for m in FAST if ((df.backbone==bb)&(df.method==m)).any()]
    for t in TR:
        vals={}
        for m in ms:
            g=df[(df.backbone==bb)&(df.method==m)].sort_values("img_idx")
            vals[m]=np.array([r["stability"][t]["pearson"] for r in g.to_dict("records")])
        pairs=[]
        for a,b in itertools.combinations(ms,2):
            try: _,p=wilcoxon(vals[a],vals[b])
            except ValueError: p=1.0
            pairs.append([a,b,float(np.mean(vals[a])-np.mean(vals[b])),p])
        adj=holm(pairs)
        for i,(a,b,d,p) in enumerate(pairs):
            rows.append({"backbone":bb,"transform":t,"a":a,"b":b,"diff":round(d,3),
                         "p_raw":round(p,4),"p_holm":round(float(adj[i]),4),
                         "sig":bool(adj[i]<0.05)})
pt=pd.DataFrame(rows); pt.to_csv(f"{RES}/tables/holm_per_transform.csv",index=False)

print("=== Per-transform: is Grad-CAM significantly LESS stable than each other method? (Holm<0.05) ===")
for bb in ["resnet50","efficientnet_b0"]:
    for t in TR:
        sub=pt[(pt.backbone==bb)&(pt.transform==t)&((pt.a=="GradCAM")|(pt.b=="GradCAM"))]
        sig=[]
        for _,r in sub.iterrows():
            other=r["b"] if r["a"]=="GradCAM" else r["a"]
            # diff is a-b; GradCAM less stable => (GradCAM - other)<0 and sig
            gc_minus = r["diff"] if r["a"]=="GradCAM" else -r["diff"]
            if r["sig"] and gc_minus<0: sig.append(other)
        print(f"  {bb:16s} {t:11s}: GradCAM < {sig if sig else 'none (n.s.)'}")

# ---------- (b) CI-augmented table values ----------
def boot(vals):
    vals=np.asarray(vals,float)
    b=[rng.choice(vals,len(vals),replace=True).mean() for _ in range(2000)]
    return vals.mean(), np.percentile(b,2.5), np.percentile(b,97.5)

def per_img(g,metric):
    return np.array([np.mean([r["stability"][t][metric] for t in TR]) for r in g.to_dict("records")])

lines=[]
for bb in ["resnet50","efficientnet_b0"]:
    for m in ["GradCAM","GradCAM++","ScoreCAM","EigenCAM","LayerCAM"]:
        g=df[(df.backbone==bb)&(df.method==m)]
        if len(g)==0: continue
        pe=boot(per_img(g,"pearson")); ss=boot(per_img(g,"ssim")); io=boot(per_img(g,"iou"))
        lines.append(f"{bb} & {m} & {pe[0]:.3f} [{pe[1]:.3f},{pe[2]:.3f}] & "
                     f"{ss[0]:.3f} [{ss[1]:.3f},{ss[2]:.3f}] & {io[0]:.3f} [{io[1]:.3f},{io[2]:.3f}] \\\\")
open(f"{RES}/tables/stability_ci_rows.tex","w").write("\n".join(lines))
print("\n=== Stability table rows WITH 95% CI (also saved to stability_ci_rows.tex) ===")
print("\n".join(lines))

# faithfulness CI rows
fl=[]
for bb in ["resnet50","efficientnet_b0"]:
    for m in ["GradCAM","GradCAM++","ScoreCAM","EigenCAM","LayerCAM"]:
        g=df[(df.backbone==bb)&(df.method==m)]
        if len(g)==0: continue
        d=boot(g["deletion_auc"].values); ins=boot(g["insertion_auc"].values); ad=boot(g["average_drop"].values)
        fl.append(f"{bb} & {m} & {d[0]:.3f} [{d[1]:.3f},{d[2]:.3f}] & {ins[0]:.3f} [{ins[1]:.3f},{ins[2]:.3f}] & "
                  f"{ad[0]:.1f} [{ad[1]:.1f},{ad[2]:.1f}] \\\\")
open(f"{RES}/tables/faithfulness_ci_rows.tex","w").write("\n".join(fl))
print("\n=== Faithfulness rows WITH 95% CI (saved) ===")
print("\n".join(fl))
