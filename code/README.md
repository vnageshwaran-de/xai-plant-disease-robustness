# Reproducing the Experiments

XAI robustness benchmark for plant disease recognition (CSoNet 2026).
All results are produced on **CPU**; no GPU required. Seed = 42 throughout.

## Environment

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install datasets grad-cam scikit-image scikit-learn scipy matplotlib pandas huggingface_hub
```
Tested with Python 3.10, torch 2.13 (CPU), pytorch-grad-cam.

## Paths

Edit the path constants at the top of `common.py`, `model_utils.py`,
`xai_lib.py`, `03a/03b/04/05` if you relocate the project. They point to:
- `DATA_ROOT` — extracted PlantVillage subset
- `RES` — `results/` directory (tables, figures, raw records)
- `data.zip` — the downloaded PlantVillage archive

## Pipeline (run in order)

```bash
# 0. Download the PlantVillage archive once (2.18 GB) to DATA dir:
#    hf_hub_download('mohanty/PlantVillage','data.zip', repo_type='dataset')
#    or curl the resolve URL. Then:

python 01_prepare_data.py         # extract leaf-split subset (35 train / 25 test per class)
python 02_train.py resnet50       # extract features + train linear head + eval
python 02_train.py efficientnet_b0
python 02_train.py vit_b_16
python 03a_pred_robustness.py     # build XAI subset + prediction-consistency table
# 03b is time-budgeted & RESUMABLE — re-run until it prints ALL_DONE:
python 03b_xai_eval.py            # explanation stability + faithfulness per (backbone,method,image)
python 04_analyze.py              # aggregate, bootstrap CIs, Holm tests, Figures 1-4 + tables
python 05_qualitative.py          # Figures 5-6 (qualitative montages); run twice (skip-if-exists)
```

`02_train.py` and `03b_xai_eval.py` checkpoint within a time budget so they can be
run repeatedly on a constrained machine; each re-run resumes from the last saved
state.

## Outputs

- `results/tables/clf_*.json` — classification accuracy / macro-F1 per backbone
- `results/raw/predrob_*.json` — prediction consistency & F1 drop per transform
- `results/raw/xai_records.jsonl` — one JSON record per (backbone, method, image):
  faithfulness (deletion/insertion/avg-drop) + per-transform stability (Pearson/SSIM/IoU)
- `results/tables/stability*.csv`, `faithfulness.csv`, `holm_pairwise.csv`, `summary.json`
- `results/figures/fig1..fig6*.png`

## Files

| File | Purpose |
|------|---------|
| `common.py` | data loading, preprocessing, backbone builders |
| `model_utils.py` | assemble CAM-compatible backbone+head models |
| `xai_lib.py` | transformation protocol, stability & faithfulness metrics |
| `01_prepare_data.py` | selective subset extraction (leaf split) |
| `02_train.py` | feature extraction + linear-head training + eval |
| `03a_pred_robustness.py` | XAI subset + prediction robustness |
| `03b_xai_eval.py` | explanation stability + faithfulness (resumable) |
| `04_analyze.py` | aggregation, statistics, quantitative figures |
| `05_qualitative.py` | qualitative heatmap montages |

Every table and figure in `paper/paper.tex` is regenerated from these outputs.
