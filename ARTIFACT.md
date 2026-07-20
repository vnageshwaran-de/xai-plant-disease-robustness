# Reproducibility Artifact

**Paper:** *How Reliable Is Explainable AI for Plant Disease Recognition? A
Robustness Evaluation under Real-World Image Transformations* (CSoNet 2026).

This artifact reproduces **every table and figure** in the paper from raw data
with a single command. All computation is **CPU-only**; no GPU is required. Every
run is deterministic (global seed = 42).

---

## 1. What is included

```
XAI_PlantDisease_CSoNet2026/
├── run_all.sh              # one-command end-to-end reproduction
├── requirements.txt        # exact pinned package versions
├── ARTIFACT.md             # this file
├── decision_log.md         # assumptions, scope decisions, caveats
├── submission_checklist.md
├── code/                   # pipeline (see code/README.md)
│   ├── common.py, model_utils.py, xai_lib.py
│   ├── 01_prepare_data.py … 05_qualitative.py
│   └── README.md
├── results/                # PRE-COMPUTED outputs shipped for inspection
│   ├── tables/  (clf_*.json, stability*.csv, faithfulness.csv, holm_pairwise.csv, summary.json)
│   ├── raw/     (xai_records.jsonl — one record per backbone×method×image; predrob_*.json; xai_subset.json)
│   ├── figures/ (fig1–fig6 PNGs)
│   └── models/  (trained linear heads)
└── paper/                  # LNCS source, references.bib, paper.pdf
```

The `results/` directory is shipped **pre-computed** so reviewers can verify every
number without re-running. `run_all.sh` regenerates it from scratch.

## 2. Requirements

- Linux/macOS, Python 3.10+, ~3 GB free RAM, ~5 GB free disk.
- Internet access on first run (downloads the PlantVillage archive, ~2.18 GB,
  and ImageNet-pretrained weights).
- TeX Live (`pdflatex`, `bibtex`) only if you want to recompile the PDF.

```bash
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 3. Reproduce everything

```bash
bash run_all.sh
```

This: downloads the dataset → extracts the fixed leaf-split subset → trains the
three linear heads → measures prediction robustness → computes explanation
stability + faithfulness for every (backbone, method, image) → aggregates with
bootstrap CIs and Holm tests → renders all figures → compiles `paper/paper.pdf`.

**Approximate CPU runtime** (4 cores): dataset download 1–2 min; feature
extraction + heads ~10 min; XAI evaluation ~20 min (Score-CAM dominates); analysis
+ figures ~1 min.

### Configuration (optional environment variables)
| Variable | Default | Meaning |
|----------|---------|---------|
| `PVXAI_ROOT` | repo dir | project root |
| `PVXAI_DATA` | `ROOT/data/pv` | extracted image subset |
| `PVXAI_ZIP` | `ROOT/data/data.zip` | PlantVillage archive |
| `PVXAI_RESULTS` | `ROOT/results` | outputs |
| `PVXAI_BUDGET` | unbounded | seconds/invocation; set small to enable resumable chunks on constrained machines |

## 4. Reproduce a single result

```bash
cd code
python3 04_analyze.py          # regenerates all tables/figures from results/raw/xai_records.jsonl
```
`04_analyze.py` is idempotent: re-running it on the shipped `results/raw/` yields a
byte-identical `summary.json` (verified).

## 5. Claims → evidence map

| Paper claim | File |
|-------------|------|
| Classification accuracy (Table 1) | `results/tables/clf_*.json` |
| Prediction consistency / F1 drop (Table 2, Fig 1) | `results/raw/predrob_*.json` |
| Explanation stability (Table 3, Fig 2, Table 5) | `results/tables/stability*.csv` |
| Holm-corrected significance | `results/tables/holm_pairwise.csv` |
| Faithfulness (Table 4, Fig 3) | `results/tables/faithfulness.csv` |
| Qualitative montages (Fig 5–6) | `results/figures/fig5*,fig6*` |
| Per-image raw measurements | `results/raw/xai_records.jsonl` |

## 6. Known scope (see decision_log.md)
CPU-scale study: frozen-backbone linear probing, class-balanced subset
(35 train / 25 test per class × 38 classes), 38-image explanation set, single
transformation severities, Score-CAM on a 12-image EfficientNet-B0 subset. These
bound absolute magnitudes but not the protocol, which transfers unchanged to
larger settings.
