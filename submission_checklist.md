# CSoNet 2026 Submission Checklist

Conference: 15th Int. Conf. on Computational Science and Network Intelligence,
Nov 16–18 2026, Ho Chi Minh City. Springer LNCS. Track deadline: **July 31, 2026**.
Submit at: **https://meteor.springer.com/CSoNet2026** (⚠️ no post-submission edits).

## Before you submit

- [ ] **Add author metadata** — replace `[AUTHOR NAME]`, `[AFFILIATION]`, `[EMAIL]`
      in `paper/paper.tex` (the only placeholders). Recompile.
- [ ] **AI-assistance disclosure** — AI cannot be a listed author (Springer policy).
      Check Springer's current AI-use disclosure requirement and add a statement if
      required.
- [ ] **Verify final PDF renders** — title, abstract, all 6 figures, 5 tables,
      references, appendix. (Current build: **14 pages**, LNCS regular-paper max.)
- [ ] **Format compliance** — `llncs.cls` unmodified margins/font; bibliography
      `splncs04.bst`; **27 references** (within the 20–35 requirement); 6 keywords.
- [ ] **Spot-check bibliography** — confirm `too2019` DOI; confirm `barman2026review`
      (in-press systematic review) final volume/pages/DOI or swap it out.
- [ ] **Page limit** — must stay ≤14 including references. If author block pushes it
      over, trim Appendix Table 5 first.

## Compile

```bash
cd paper
pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```

## Package contents

```
XAI_PlantDisease_CSoNet2026/
├── paper/
│   ├── paper.tex            # LNCS source
│   ├── paper.pdf            # compiled (14 pp.)
│   ├── references.bib       # 27 verified references
│   ├── llncs.cls, splncs04.bst
│   └── figures/             # fig1–fig6 PNGs
├── code/                    # full reproducible pipeline + README
├── results/                 # tables, figures, raw per-image records
├── decision_log.md          # assumptions, scope decisions, caveats
└── submission_checklist.md  # this file
```

## Integrity notes

- All reported numbers are computed from executed code on real PlantVillage data
  (`results/raw/`). None are estimated.
- All citations were located and opened during preparation.
- Scope reductions (CPU-scale linear probing, subsets, reduced Score-CAM) are
  disclosed in the paper and `decision_log.md`.
