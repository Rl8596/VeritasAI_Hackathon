# Plan

## The idea in one sentence
Our project helps **a clinician in a small clinic without a radiologist** by taking **a chest X-ray** and giving back
**a pneumonia flag, a heatmap of where the model looked, and a warning when the patient is unlike the people the model learned from**.

## Why this is our own question (not the tutorial)
The standard pneumonia tutorial dataset (Kermany et al. 2018) contains only children aged 1 to 5 from one hospital in China.
Most tutorials report 90%+ accuracy on its test set and stop there. We ask what the brief does not:

1. **Does a children-only model still work on adults from another hospital?** (RSNA 2018 data, USA, with age, sex and view position.)
2. **Why not?** Which shortcuts does it use, and can a simple classifier tell the two hospitals apart?
3. **Can the model know when it is out of its depth?** An out-of-distribution warning.
4. **How many adult X-rays does it take to fix it?** A learning curve.

## Track fit
AI + Health: "Help one specific person catch a health problem sooner, or manage one better."

## What gets built
| # | Piece | Where | Status |
|---|---|---|---|
| 1 | Data preparation for both datasets, patient-level splits | `src/pneumoshift/data.py`, `scripts/01_prepare_data.py` | done |
| 2 | Core library: model, Grad-CAM, OOD warning, metrics | `src/pneumoshift/` | done |
| 3 | Children-only model (the "tutorial" model) | `scripts/02_train.py --name kids_only` | training |
| 4 | Data audit (duplicates, shortcuts) and simple baselines | `scripts/03_audit.py`, `scripts/04_baselines.py` | in progress |
| 5 | Shift analysis: children vs adults, subgroups, OOD warning, learning curve, heatmap check | `scripts/05_*.py` | next |
| 6 | Children + adults model (our fix) | `scripts/02_train.py --name kids_plus_adults` | next |
| 7 | Demo app (Gradio on Hugging Face Spaces), models on the Hugging Face Hub | `app/` | in progress |
| 8 | Colab notebook that runs the whole thing | `notebooks/` | later |
| 9 | One-page write-up, 2-minute video script | `README.md`, `docs/` | later |

## Rules we follow so the numbers can be trusted
- Test data is never used for training, for choosing thresholds, or for choosing between models.
- Splits are by patient, not by image.
- Every result is reported next to a simple baseline.
- The decision threshold is fixed on validation data (95% sensitivity) and then kept the same for every group.
- We show failures, not just averages.

## What the two of us must do by hand
- [ ] Record the 2-minute demo video (script will be in `docs/VIDEO_SCRIPT.md`).
- [ ] Open the Colab notebook and the Space in a logged-out browser to check that judges can open them.
- [ ] Fill in the submission form (team names, links, what we added beyond the brief).
- [ ] Read `docs/` so both of us can answer questions about every number in the finals.
