# How Reliable Is Explainable AI for Plant Disease Recognition?

**A Robustness Evaluation under Real-World Image Transformations** — companion code and paper (CSoNet 2026).

Benchmarking the robustness of CAM-based explanations (Grad-CAM, Grad-CAM++, Score-CAM,
Eigen-CAM, Layer-CAM) for plant-disease CNNs under realistic, label-preserving image
perturbations (blur, rotation, brightness, noise) on the leaf-split PlantVillage dataset.

## Key findings
- **Perturbation-dependent fragility:** brightness is benign; blur and noise strongly
  destabilise both predictions and explanations.
- **Stability ≠ faithfulness:** Grad-CAM is the least stable yet most faithful; Layer-CAM
  and Grad-CAM++ are the most stable.
- **Fragility worsens toward deployment accuracy:** on a fully fine-tuned **98.9%** model,
  Grad-CAM and Score-CAM degrade sharply while element-wise / principal-component maps stay robust.
- **Stability can be gamed:** the "most stable" methods become near-class-agnostic at
  deployment accuracy (a saliency sanity-check failure) — so stability must be read jointly
  with faithfulness.

## Repository layout
```
paper/     LNCS LaTeX source, references.bib, compiled paper.pdf, figures
code/      full reproducible pipeline (01–16 scripts) + README
results/   tables, figures, raw per-image records (model weights are gitignored)
colab_finetune_experiments.ipynb   GPU fine-tuning + extended analysis
run_all.sh, requirements.txt        one-command reproduction
ARTIFACT.md, decision_log.md        reviewer guide + design decisions
```

## Reproduce
```bash
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
bash run_all.sh          # downloads data, trains, evaluates, compiles the paper
```
See `ARTIFACT.md` for details and a claims→evidence map. Model weights are not tracked in
git (48 MB); they regenerate from `code/02_train.py`, `code/09_finetune.py`, or the notebook.
