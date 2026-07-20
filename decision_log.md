# Decision Log — CSoNet 2026 Paper
### "How Reliable Is Explainable AI for Plant Disease Recognition? A Robustness Evaluation under Real-World Image Transformations"

This log records the assumptions, design choices, and scope decisions made while
building the paper autonomously, plus items the author should double-check before
submission. Every reported number traces to a saved output under `results/`.

---

## 1. Interpretation of the brief

- **Input.** A title + a detailed section-by-section outline (no abstract). The
  abstract was written from scratch to match the outline and the executed study.
- **Best-fit track.** **C (AI / data-driven)** with strong overlap on
  **D (applications, agriculture)**. The contribution is an evaluation framework +
  empirical study, not a new model.
- **Claimed contributions honored:** (i) reproducible perturbation protocol,
  (ii) benchmarking framework comparing XAI methods, (iii) empirical study,
  (iv) code/data release.

## 2. Dataset

- **Choice (given by author):** PlantVillage **color** config via the official
  Hugging Face repo `mohanty/PlantVillage`, using the repo's **leaf-based**
  train/test split (prevents leakage across augmented views of one leaf).
- **Subset (scope):** fixed-seed (42) class-balanced subset — **35 train / 25 test
  per class**, all **38 classes** (1,330 train / 950 test images). Chosen so the
  full pipeline runs at CPU scale in this environment. Documented in the paper.

## 3. Models / training

- **Backbones:** ResNet-50, EfficientNet-B0, ViT-B/16 (ImageNet-pretrained).
- **Training = linear probing.** Backbones frozen; a linear head trained on pooled
  features (GAP for CNNs, CLS token for ViT), Adam, 300 steps on cached features.
  - *Why:* 3.8 GB RAM / CPU-only made full fine-tuning impractical. Freezing does
    **not** affect CAM computation (gradients still flow through the conv stack).
  - *Consequence:* accuracies (86.8 / 88.3 / 91.9%) are lower than the ~99% usually
    quoted for full fine-tuning on all of PlantVillage — expected and stated in the
    paper. They are more than adequate as a substrate for a robustness study.
- ViT uses native 224×224 input; CNNs use 160×160.

## 4. XAI methods and the ScoreCAM scope reduction

- **Methods:** Grad-CAM, Grad-CAM++, Eigen-CAM, Layer-CAM on **both** CNN backbones.
- **Score-CAM** performs one forward pass per activation channel (1,280 for
  EfficientNet-B0, 2,048 for ResNet-50) → ~50–90× slower than gradient methods on
  CPU (measured: ~5.8 s/img EfficientNet, ~9 s/img ResNet-50, vs 0.03–0.18 s for the
  others). **Decision:** evaluate Score-CAM on **EfficientNet-B0 only**, on a
  **12-image subset**, reported with its own sample size and wider CIs. Documented
  in the paper (Sec. 4) and reflected in every table.
- **ViT + CAM** intentionally left to future work (transformer CAMs need
  reshape-transforms and are costly on CPU); ViT appears in the classification
  table only. Stated in the paper.

## 5. Transformation protocol

Label-preserving, single severity each (applied on [0,1] image before ImageNet norm):
- **Gaussian blur** (9×9, σ=2), **Rotation** (+15°, bilinear),
  **Brightness** (×1.3, clipped), **Gaussian noise** (σ=0.08). Identity = reference.
- Geometric alignment: for rotation, the *original* CAM is rotated into the
  transformed frame and metrics are computed over the valid (non-empty) region.

## 6. Metrics & statistics

- **Prediction level:** Prediction Consistency (Eq. 1), accuracy/macro-F1 drop —
  on a 190-image set (5/class).
- **Explanation stability:** Pearson r, SSIM, top-20% IoU between aligned original
  and perturbed maps — on a 38-image set (1 correctly-classified image/class).
- **Faithfulness:** Deletion AUC, Insertion AUC (20 steps), Average Drop (%).
- **Statistics:** bootstrap 95% CIs (2,000 resamples); pairwise method comparison
  via Wilcoxon signed-rank on per-image mean Pearson + **Holm** correction.

## 7. Key findings (all real, from `results/`)

- Brightness is nearly inert (PC≈0.95, r>0.95); **blur and noise** dominate
  instability. Backbone-dependent: ResNet-50 worst under **noise**, EfficientNet-B0
  worst under **blur**.
- **Layer-CAM ≈ Grad-CAM++** most stable (r≈0.87–0.88); **Grad-CAM least stable**
  (r≈0.77), significant under Holm (p<0.05).
- **Stability ≠ faithfulness:** Grad-CAM is the *least stable* yet *most faithful*
  (lowest average drop, e.g. 19.9 on EfficientNet-B0). Score-CAM best insertion AUC.
  This trade-off is reported honestly, including where methods lose.

## 8. References

- 27 references, all located/opened during preparation (arXiv / Springer / IEEE /
  ScienceDirect / ACM / conference pages) — see `paper/references.bib`.
- Two author self-citations added at the author's request from the provided Google
  Scholar profile (Vinoth Nageshwaran): `nageshwaran2026interpretable`,
  `nageshwaran2024braintumor`, cited in the Introduction.

## 9. ⚠️ Items to double-check before submitting

1. **Author metadata.** Replace `[AUTHOR NAME]`, `[AFFILIATION]`, `[EMAIL]` (the
   only placeholders in the paper).
2. **Springer AI-disclosure policy.** AI cannot be a listed author; check Springer's
   current AI-assistance disclosure requirement and add a disclosure if required.
3. **A few bibliographic fields to verify** (located via search, verify exact
   values for camera-ready): DOI of `too2019` (Comput. Electron. Agric. 161);
   `barman2026review` is a 2026 systematic review marked *in press* — confirm final
   volume/pages/DOI or replace if not yet citable; page numbers for a few
   conference entries.
4. **Page limit.** Compiles to exactly **14 pages** (the LNCS regular-paper max,
   including references + a 1-page appendix). If the camera-ready adds author block
   content and pushes over 14, trim the appendix (Table 5) first.
5. **Meteor submission.** Submit at https://meteor.springer.com/CSoNet2026. Meteor
   allows **no post-submission edits** — verify the final PDF first.

## 10. Environment note (transparency)

While rendering a page-preview image, a set of unrelated files (a 12-page IoT
"agentic LLM" paper render) appeared in the temporary sandbox `/tmp` and were
write-protected (not created by this project). The final `paper.pdf` was verified
**independently** via `pdftotext` and a fresh md5-matched render to confirm it
contains only this plant-disease XAI paper. No injected content entered the
deliverables. Flagging for awareness only.

## 11. Fine-tuned validation (added in response to peer review, run on Colab GPU)

Reviewer concern #2 (frozen backbone) was addressed with a real experiment. An
EfficientNet-B0 was **fully fine-tuned end-to-end** (Colab T4; also reproduced at
smaller scale on CPU). Results:

- **Accuracy 98.9% / macro-F1 98.9%** on the 950-image test set (vs 88.3% linear
  probe) — the near-saturated regime of deployed models, closing the accuracy gap.
- Explanation stability re-run on n=100, all five CAM methods. **The stable-method
  ranking transfers** (Eigen-CAM 0.852, Layer-CAM 0.840, Grad-CAM++ 0.832), but
  **Grad-CAM collapses (0.771→0.197) and Score-CAM (0.761→0.044)** — fine-tuning
  *accentuates* their fragility. CIs are tight and non-overlapping at n=100.
- Faithfulness measured on clean AND perturbed inputs (#6): deletion/insertion AUCs
  shift materially under perturbation (records in `results/finetuned/`).

Reported in the paper as Section 5.6 + Table 5. Artifacts: `results/finetuned/`
(records, model, figure) and `colab_finetune_experiments.ipynb`.

## Review-response summary (what changed)
- #1 field validity: claims softened to hypotheses + lab-vs-field caveat + accuracy reconciliation.
- #2 fine-tuning: **done** (Sec 5.6, 98.9% model).
- #3 CIs/sample size: CIs added to Tables 3-4; per-transform Holm shows no single-transform
  significance at n=38 (honesty correction) — but fine-tuned n=100 CIs are decisive.
- #4 ViT: reframed CNN-only for the explanation analysis.
- #5 class-conditioning: justified (Eigen/Score-CAM class-agnostic caveat).
- #6 faithfulness under perturbation: added via the fine-tuned run.
- minors: k-sensitivity check (ranking invariant), diseased-leaf qualitative figure,
  prediction-change handling clarified, ref [3] DOI fixed.
- self-citations [17][18]: KEPT at author request, reframed for relevance.

## 12. Round-2 review response (§5.6 deepened)
- **Score-CAM r=0.044 diagnosed (not a bug):** library weights channels by softmax over
  per-channel target logits; on the 98.9% model the weights peak (entropy 6.25->2.16,
  one channel = 53% weight vs 1% frozen) -> map collapses to one volatile channel.
  Min-max weighting restores stability (0.28->0.64). Pre-softmax logit already used,
  so distinct from softmax-gradient saturation. (results/finetuned/scorecam_mechanism.json)
- **Class-sensitivity control:** class-sensitivity vs stability anti-correlated r=-0.999
  across the 4 fast methods (Eigen 0.00, GradCAM++ 0.03, Layer 0.02, GradCAM 0.97) ->
  stability can be inflated by class-insensitivity -> must read jointly with faithfulness.
  (results/finetuned/controls_summary.json)
- **Paired comparison:** Table 5 now uses the SAME 38 images for frozen vs fine-tuned,
  CIs on both columns (GradCAM 0.771->0.392 paired; 0.20 on n=100).
- **Faithfulness under perturbation:** new Table 6 (clean vs perturbed del/ins AUC).
- **Per-transform Holm corrected:** GradCAM significantly less stable in 16/24 per-transform
  comparisons (p<1e-3 noise/blur); earlier "none significant" note was a bug and is fixed.
- Abstract: added 4th finding (fragility worsens toward deployment accuracy); hedging
  harmonized with conclusion. Conclusion "confirms all three" narrowed to the stability half.
- Trimmed Fig 1 (pred), Fig 2 (stability), Fig 3 (faithfulness) — numbers are in tables;
  one figure kept (diseased-leaf qualitative montage). Still 14 pages.
Scripts: 10_scorecam_diag.py, 11_scorecam_fix.py, 12_ft_controls.py.

## 13. Round-3 review response
- **Table 6 insertion was a NOTEBOOK BUG** (insertion ~0.11). Trusted xai_lib recompute:
  clean insertion 0.39-0.43 > deletion 0.29 (correct for a 98.9% model). Table 6 replaced
  with **confidence-normalized deletion** (clean vs perturbed): 0.29 -> 0.84-1.35, i.e.
  faithfulness degrades under perturbation even after removing the base-rate confound.
  (results/finetuned/faith_norm.jsonl; scripts 13_faith_diag.py, 14_faith_recompute.py)
- Dropped r=-0.999 (n=4, not meaningful); state the pattern instead.
- Added Adebayo [1] framing: GradCAM++/LayerCAM near-class-agnostic on the deployment
  model = failure of the class-sensitivity sanity check (own sentence, cited).
- Control 1 reframed: softmax-over-channel-logits is the PUBLISHED Score-CAM spec (artifact
  of the method, not a bug); min-max = proposed MODIFICATION; held-out diagnostic (n=4) baseline
  explained (0.28 easier subset vs 0.04 n=100 headline).
- Table 5: dropped backwards arrow; removed Score-CAM row (n=100 not paired -> Control 1);
  caption now "same 38 images"; class-sensitivity labeled "higher = more class-conditional".
- Conclusion significance made backbone-accurate (Eigen-CAM only significant on ResNet-50).
- Abstract "three findings" -> "four findings".
- Restored Appendix A per-transform table (source for 0.67/0.39 and 16/24); defined the 24
  (2 backbones x 4 transforms x 3 pairs, Score-CAM excluded).
- Discussion recommendation qualified: prefer Layer/GradCAM++ only if they pass a
  class-sensitivity check on the deployed model.
- Limitations: class-sensitivity is single-architecture; Score-CAM tied to one library impl.
- Contributions bullet reworded. Dropped Table 1 (classification, folded to text) and shrank
  qualitative montage to 3 methods to hold 14 pages.
- OUTSTANDING (author to verify): ref barman2026 still in-press (no DOI); self-citations
  [17]/[18] kept at author request; no code URL in paper so no double-blind issue.

## 14. Round-4 review response (accept w/ minor revisions)
- Table 5 (faith-pert): added Insertion columns (clean/pert); caption+text now EXPLAIN the
  >1 normalized values (removing salient pixels raises confidence above the perturbed
  baseline -> saliency anti-correlated with evidence) and the n=24 (cost: ~30 fwd passes
  per image/method/condition). "roughly triples" -> "2.9-4.6x".
- **Self-consistency control (NEW, script 16):** 1-r between CAM(x) and CAM(x+jitter,
  same class) = ~0.01 for ALL methods incl Grad-CAM, vs class-sensitivity 0.00-0.97.
  Proves Grad-CAM's 0.97 is genuine class-discrimination, not noise. Specified class-
  sensitivity = single random class, seed 42.
- §5.3 slip fixed: only Score-CAM foreshadows the collapse; Eigen-CAM is MOST stable when FT.
- Conclusion "simple fix" -> "candidate fix" (agrees with §5.6 "proposed modification").
- Table 4 frozen CIs harmonized to match Table 2 exactly (same bootstrap values).
- Appendix note no longer claims to hold p-values (it's means); p-values -> released records.
- "montage in the artifact" -> "released records". Contributions bullet split into two.
- Limitations: perturbed-faithfulness + controls are single-architecture, small-n (24-38).
- Page fit: dropped classification Table 1 (inline), frozen faithfulness table -> means-only,
  DenseNet ref removed (26 refs), appendix table -> EfficientNet-B0 only and moved before
  bibliography (into body flow) to hold 14 pages.
- STILL author's call: ref barman2026 in-press (no DOI); self-citations [17]/[18] kept.

## 15. Round-5 review response (submission-ready)
- Table 5 insertion arrow fixed: removed the up-arrow from BOTH perturbed columns;
  caption now states values >1 are a degeneracy signal (masked input beats full input),
  not a faithfulness gain, so arrows apply only to the clean columns.
- CI-claim regression fixed: Statistics paragraph now says CIs are printed in Tables 2 & 4
  and provided in the released records for the rest (page-limit constraint stated honestly).
  Generated results/tables/ci_supplement.csv (CIs for Table 5 faith-pert and Table 1 pred-
  consistency); Tables 3 & 6 CIs already in faithfulness.csv / stability.csv. Claim now true.
- Free win added: Grad-CAM is the ONLY method with perturbed deletion <1 (0.84 vs 1.19-1.35),
  so it remains the most faithful under perturbation -> extends the trade-off to the
  decision-relevant perturbed regime.
- Page fit: removed LIME and Integrated Gradients citations (singly-cited "complementary
  family"); 24 references (still within 20-35). Condensed statistics + faithfulness prose.
- Leftovers (author's call, unchanged): ref barman2026 in-press DOI; self-citations kept;
  class-sensitivity single random-class draw (seed 42 stated; averaging optional).

## 16. Ref [3] resolved (was "in press")
Verified via Crossref: the systematic-review reference was formerly a bare "in press"
entry AND had the WRONG authors (I had guessed "Barman, U."). Corrected to the real
record: Hamdaoui W., Workneh A.D., El Hilali Alaoui A., Elmouhtadi M., "Machine learning
and explainable artificial intelligence for plant disease recognition...", Computers and
Electrical Engineering 134, 111130 (2026), doi:10.1016/j.compeleceng.2026.111130. Key
renamed barman2026review -> hamdaoui2026review. No fabricated data.

## 17. Author metadata COMPLETE (correction)
The author block is filled in (no placeholders remain): Pham Tien Phuc (FPT, corresponding
inst 1), Vinoth Nageshwaran (Univ. of the Cumberlands, inst 2), and Quan Thanh Tho, Duong
Phu Bao, Ngo Trung Tin, Nguyen Huy, Duong Tan Phuc, Le Tran Anh Thuong (HCMUT, inst 3),
with all emails. The earlier "replace [AUTHOR NAME]" checklist item (Sec. 9.1) is DONE and
superseded. Names romanized to ASCII (no vntex/Vietnamese font in the build); verify order
and add diacritics at camera-ready via XeLaTeX/Overleaf if desired.
Remaining before submission: Springer AI-disclosure policy check only.
