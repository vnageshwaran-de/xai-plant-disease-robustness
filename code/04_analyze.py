"""
04_analyze.py
Aggregate XAI records into tables with bootstrap 95% CIs, run Holm-corrected
pairwise method comparisons, and render quantitative figures.
"""
import os, json, itertools
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as C
RES = C.RESULTS
FIG = RES + "/figures"; TAB = RES + "/tables"
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
rng = np.random.default_rng(42)
TRANSFORMS = ["blur", "rotate", "brightness", "noise"]
METHOD_ORDER = ["GradCAM", "GradCAM++", "ScoreCAM", "EigenCAM", "LayerCAM"]

recs = [json.loads(l) for l in open(f"{RES}/raw/xai_records.jsonl")]
df = pd.DataFrame(recs)

def boot_ci(vals, nboot=2000):
    vals = np.asarray(vals, float)
    if len(vals) == 0: return (np.nan, np.nan, np.nan)
    means = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(nboot)]
    return float(vals.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

# ---------- Stability table (per backbone, method, transform, metric) ----------
rows = []
for (bb, m), g in df.groupby(["backbone", "method"]):
    for t in TRANSFORMS:
        for metric in ["pearson", "ssim", "iou"]:
            vals = [r["stability"][t][metric] for r in g.to_dict("records")]
            mean, lo, hi = boot_ci(vals)
            rows.append({"backbone": bb, "method": m, "transform": t, "metric": metric,
                         "n": len(vals), "mean": mean, "ci_lo": lo, "ci_hi": hi})
stab = pd.DataFrame(rows)
stab.to_csv(f"{TAB}/stability.csv", index=False)

# per-image mean stability (avg over transforms) for stats + overall score
def per_image_mean(g, metric):
    return np.array([np.mean([r["stability"][t][metric] for t in TRANSFORMS])
                     for r in g.to_dict("records")])

overall = []
for (bb, m), g in df.groupby(["backbone", "method"]):
    for metric in ["pearson", "ssim", "iou"]:
        vals = per_image_mean(g, metric)
        mean, lo, hi = boot_ci(vals)
        overall.append({"backbone": bb, "method": m, "metric": metric, "n": len(vals),
                        "mean": mean, "ci_lo": lo, "ci_hi": hi})
pd.DataFrame(overall).to_csv(f"{TAB}/stability_overall.csv", index=False)

# ---------- Faithfulness table ----------
frows = []
for (bb, m), g in df.groupby(["backbone", "method"]):
    for metric in ["deletion_auc", "insertion_auc", "average_drop"]:
        mean, lo, hi = boot_ci(g[metric].values)
        frows.append({"backbone": bb, "method": m, "metric": metric, "n": len(g),
                      "mean": mean, "ci_lo": lo, "ci_hi": hi})
faith = pd.DataFrame(frows)
faith.to_csv(f"{TAB}/faithfulness.csv", index=False)

# ---------- Holm-corrected pairwise comparisons (per backbone) ----------
# Primary metric: per-image mean Pearson stability. Paired Wilcoxon on the shared
# 38-image set (fast methods). ScoreCAM excluded (n=12, not paired-complete).
holm_rows = []
for bb in ["resnet50", "efficientnet_b0"]:
    methods = [m for m in ["GradCAM", "GradCAM++", "EigenCAM", "LayerCAM"]
               if ((df.backbone == bb) & (df.method == m)).any()]
    per = {}
    for m in methods:
        g = df[(df.backbone == bb) & (df.method == m)].sort_values("img_idx")
        per[m] = np.array([np.mean([r["stability"][t]["pearson"] for t in TRANSFORMS])
                           for r in g.to_dict("records")])
    pvals = []
    for a, b in itertools.combinations(methods, 2):
        try:
            stat, p = wilcoxon(per[a], per[b])
        except ValueError:
            p = 1.0
        pvals.append([bb, a, b, float(np.mean(per[a]) - np.mean(per[b])), p])
    # Holm correction
    order = np.argsort([x[4] for x in pvals]); mtests = len(pvals)
    adj = [None] * mtests; running = 0.0
    for rank, idx in enumerate(order):
        a = (mtests - rank) * pvals[idx][4]
        running = max(running, a); adj[idx] = min(1.0, running)
    for i, row in enumerate(pvals):
        holm_rows.append({"backbone": row[0], "method_a": row[1], "method_b": row[2],
                          "mean_diff_pearson": row[3], "p_raw": row[4],
                          "p_holm": float(adj[i]), "sig_holm_0.05": bool(adj[i] < 0.05)})
pd.DataFrame(holm_rows).to_csv(f"{TAB}/holm_pairwise.csv", index=False)

# ---------- Summary JSON ----------
summary = {"n_records": len(df),
           "stability_overall": overall,
           "faithfulness": frows,
           "holm": holm_rows}
json.dump(summary, open(f"{TAB}/summary.json", "w"), indent=1)

# ================= FIGURES =================
COL = {"GradCAM": "#1f77b4", "GradCAM++": "#ff7f0e", "ScoreCAM": "#2ca02c",
       "EigenCAM": "#d62728", "LayerCAM": "#9467bd"}

# Fig 1: prediction robustness (PC + macro-F1 drop) per transform, per backbone
pred = {bb: json.load(open(f"{RES}/raw/predrob_{bb}.json")) for bb in ["resnet50", "efficientnet_b0"]}
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
conds = ["blur", "rotate", "brightness", "noise"]
x = np.arange(len(conds)); w = 0.35
for ax, key, ttl in [(axes[0], "prediction_consistency", "Prediction consistency"),
                     (axes[1], "macro_f1_drop", "Macro-F1 drop")]:
    for j, bb in enumerate(["resnet50", "efficientnet_b0"]):
        vals = [pred[bb]["conditions"][c][key] for c in conds]
        ax.bar(x + (j - 0.5) * w, vals, w, label=bb.replace("_", "-"),
               color=["#4C72B0", "#DD8452"][j])
    ax.set_xticks(x); ax.set_xticklabels(conds, rotation=15); ax.set_title(ttl)
    ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{FIG}/fig1_prediction_robustness.png", dpi=150); plt.close()

# Fig 2: explanation stability (mean Pearson) per method across transforms, per backbone
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, bb in zip(axes, ["resnet50", "efficientnet_b0"]):
    ms = [m for m in METHOD_ORDER if ((stab.backbone == bb) & (stab.method == m)).any()]
    x = np.arange(len(TRANSFORMS)); w = 0.8 / len(ms)
    for k, m in enumerate(ms):
        sub = stab[(stab.backbone == bb) & (stab.method == m) & (stab.metric == "pearson")]
        sub = sub.set_index("transform").reindex(TRANSFORMS)
        err = [sub["mean"] - sub["ci_lo"], sub["ci_hi"] - sub["mean"]]
        ax.bar(x + (k - len(ms)/2 + 0.5) * w, sub["mean"], w, yerr=err, capsize=2,
               label=m, color=COL[m])
    ax.set_xticks(x); ax.set_xticklabels(TRANSFORMS, rotation=15)
    ax.set_title(f"{bb.replace('_','-')}"); ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="k", lw=0.6)
axes[0].set_ylabel("Explanation stability (Pearson r)")
axes[1].legend(fontsize=7, ncol=1)
plt.tight_layout(); plt.savefig(f"{FIG}/fig2_stability_pearson.png", dpi=150); plt.close()

# Fig 3: faithfulness deletion vs insertion (lower del better, higher ins better)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
for ax, bb in zip(axes, ["resnet50", "efficientnet_b0"]):
    for m in [x for x in METHOD_ORDER if ((faith.backbone==bb)&(faith.method==x)).any()]:
        d = faith[(faith.backbone==bb)&(faith.method==m)&(faith.metric=="deletion_auc")]["mean"].values[0]
        ins = faith[(faith.backbone==bb)&(faith.method==m)&(faith.metric=="insertion_auc")]["mean"].values[0]
        ax.scatter(d, ins, s=70, color=COL[m], label=m, edgecolor="k", zorder=3)
    ax.set_xlabel("Deletion AUC (lower better)"); ax.set_ylabel("Insertion AUC (higher better)")
    ax.set_title(bb.replace("_","-")); ax.grid(alpha=0.3); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{FIG}/fig3_faithfulness.png", dpi=150); plt.close()

# Fig 4: stability distribution (boxplot of per-image mean Pearson) efficientnet
fig, ax = plt.subplots(figsize=(6.5, 3.6))
bb = "efficientnet_b0"
ms = [m for m in METHOD_ORDER if ((df.backbone==bb)&(df.method==m)).any()]
data = [per_image_mean(df[(df.backbone==bb)&(df.method==m)], "pearson") for m in ms]
bp = ax.boxplot(data, labels=ms, patch_artist=True, showmeans=True)
for patch, m in zip(bp["boxes"], ms): patch.set_facecolor(COL[m]); patch.set_alpha(0.6)
ax.set_ylabel("Per-image mean stability (Pearson r)")
ax.set_title("Explanation stability distribution (EfficientNet-B0)")
ax.grid(axis="y", alpha=0.3); plt.xticks(rotation=15)
plt.tight_layout(); plt.savefig(f"{FIG}/fig4_stability_box.png", dpi=150); plt.close()

print("Analysis complete. Figures + tables written.")
print("\n=== Overall stability (mean Pearson across transforms) ===")
for o in overall:
    if o["metric"] == "pearson":
        print(f"  {o['backbone']:16s} {o['method']:10s} n={o['n']:2d} "
              f"r={o['mean']:.3f} [{o['ci_lo']:.3f},{o['ci_hi']:.3f}]")
print("\n=== Holm-significant pairs (p<0.05) ===")
for h in holm_rows:
    if h["sig_holm_0.05"]:
        print(f"  {h['backbone']:16s} {h['method_a']} vs {h['method_b']} "
              f"diff={h['mean_diff_pearson']:+.3f} p_holm={h['p_holm']:.4f}")
