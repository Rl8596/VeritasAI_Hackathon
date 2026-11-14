# PneumoShift: handoff for the next Claude session

You are taking over a hackathon project from another Claude session. This file is everything you need.
Read it fully, then `docs/HACKATHON_CONTEXT.md` (rules, judging) and skim `docs/RESEARCH_NOTES.md` (sources, citations).

Written Sat 14 Nov 2026, 11:50 UTC. **Deadline: Sun 15 Nov 2026, 11:59 PM ET (= Mon 04:59 UTC).**

## 0. Rules from the team (follow them)

- The team is two high-school students. They write German. Answer in German, short and concrete (Swiss spelling: "ss", not "ß").
- Usage is limited. Few tool calls, no subagents unless they ask, short answers.
- Goal: win the Veritas AI International High School Hackathon 2026, track **AI + Health**, Advanced level.
- A scam-SMS project is forbidden (it is the organisers' own demo submission).
- Hugging Face must be used. Models are published under the account `RL8596`.
- Never ask for a token in chat. Uploading to `RL8596` needs that account's write token, entered through the secure credential form (host `huggingface.co`). Downloading needs no token.
- No pull request and no push to `main` unless the team asks. Commit and push your work often.
- Judging round 1 is blind: no names or school in the repo. Names go into the submission form only.
- Check the hackathon rules on AI assistance and tell the team to disclose it as the rules require. This file may be deleted before submission if they wish.

## 1. The project in one sentence

It helps **a clinician in a small clinic without a radiologist** by taking **a chest X-ray** and giving back
**a pneumonia flag, a heatmap of where the model looked, and a warning when the patient is unlike the people the model learned from**.

Our own question (not the tutorial): the standard pneumonia tutorial dataset holds only children aged 1 to 5 from one hospital.
**What happens when the tutorial model meets adults from another hospital, and can the tool notice it?**

Data (both from the Hugging Face Hub, nothing is stored in git):

| | Children | Adults |
|---|---|---|
| Source | Kermany et al. 2018, Guangzhou, CC BY 4.0, `hf-vision/chest-xray-pneumonia` | RSNA Pneumonia Detection Challenge 2018 (NIH Clinical Center, USA), `kellly/RSNA-Pneumonia-Detection` |
| Size | 5,856 X-rays | 26,684 exams of 11,452 patients, with age, sex, view (AP = bedside, PA = standing), radiologist boxes |
| Our split (by patient) | train 4,578 / val 654 / official test 624 | adapt 16,010 / val 2,667 / test 8,007 (7,694 adults) |

Adult labels: "Normal", "No Lung Opacity / Not Normal" (sick with something else), "Lung Opacity".
**"Lung Opacity" is a radiologist's finding, not a confirmed pneumonia diagnosis. Say so wherever it matters.**
We report adults twice: lung opacity vs normal (easy) and lung opacity vs everything else (realistic).

## 2. Set up a fresh clone

```bash
pip install -r requirements.txt
python scripts/01_prepare_data.py     # downloads about 5 GB, writes data_cache/ (about 15 min)
python scripts/hub_models.py pull     # trained models and per-image predictions -> artifacts/
```

Pipeline, in order (each script's docstring explains it):

| Step | Command | Output |
|---|---|---|
| Data | `scripts/01_prepare_data.py` | `data_cache/processed/` |
| Train | `scripts/02_train.py --name kids_only` | `artifacts/kids_only/`, `results/headline_kids_only.json` |
| | `scripts/02_train.py --name kids_plus_adults --rsna-n 6000` | same for the comparison model |
| Audit | `scripts/03_audit.py`, `scripts/04_baselines.py` | `results/audit.json`, `results/baselines.json` |
| Shift analysis | `scripts/05_shift_analysis.py --model <name>` | `results/shift_<name>.json`, 5 figures, `ood.npz`, metrics into the model's `config.json` |
| Heatmaps | `scripts/06_heatmap_check.py --models kids_only kids_plus_adults` | `results/heatmap_check.json`, figures |
| Fine-tuning curve | `scripts/07_finetune_curve.py` | `results/finetune_curve.json`, figure |
| App | `python app/app.py` (port 7860) | Gradio demo, loads models from the Hub |
| Hub | `scripts/hub_models.py push <name>`, `scripts/deploy_space.py` | needs the write token |
| Notebook | `python scripts/make_notebook.py` | `notebooks/pneumoshift_colab.ipynb` |

Training one model takes about 1 hour on 4 CPU cores and a few minutes on a GPU. Library code is in `src/pneumoshift/`.
Model: ResNet-18 pretrained on ImageNet (`timm/resnet18.tv_in1k` from the Hub), fine-tuned, one output.
Threshold rule, fixed before testing: 95% sensitivity on validation data. Never tune anything on test data.

## 3. State at handoff

The newest work is always on the branch `claude/loving-hawking-jtiwb3` (later results are pushed there automatically).
If a file named here is missing in your checkout: `git fetch origin claude/loving-hawking-jtiwb3` and merge it.

Done and in git: data pipeline, library, children-only model, audit, baselines, shift analysis of the children-only model,
demo app, Colab notebook, scripts 06 and 07.

An unattended script was still running on the old machine at handoff. That machine may be gone. **Check what arrived:**

| Should exist | If it is missing |
|---|---|
| Hub: `RL8596/pneumoshift-kids-only` (uploaded, confirmed) | |
| Hub: `RL8596/pneumoshift-kids-plus-adults` | retrain (step Train), run 05, push |
| git: `results/shift_kids_plus_adults.json`, `results/heatmap_check.json` | run steps 05 and 06 |
| Hub: `...-kids-plus-300`, `...-kids-plus-1000`, `...-adults-only` | optional: `02_train.py --name kids_plus_300 --rsna-n 300`, `--name kids_plus_1000 --rsna-n 1000`, `--name adults_only --no-kids --rsna-n 6000` |
| git: `results/finetune_curve.json` | run step 07 after those models exist |

Never tested or looked at yet. Do not claim these work until you have checked:

- **No figure has been looked at by anyone.** Open every PNG in `results/figures/` and fix overlaps, cut-off labels, wrong titles.
- **Scripts 06 and 07 have never been run by a session that read their output.** Read the numbers critically.
- **The notebook has never been run in Colab.** Quick mode (`--rsna-limit 6000`, 3 epochs) is untested in every script.
- **The app was tested with one image on a desktop screen.** Phone layout untested. Known flaws: some captions are cut off, the heatmaps sit below the fold.
- **The Hugging Face Space is not online.** The upload was refused: Gradio Spaces on the free tier now need a PRO subscription.

## 4. Results so far (children-only model, held-out test patients)

Threshold 0.985, temperature-scaled scores.

| Test group | n | AUROC | Sensitivity | Specificity | Accuracy | "Always the majority class" |
|---|---|---|---|---|---|---|
| Children 1 to 5, training hospital | 624 | 0.975 | 0.979 | 0.867 | 0.937 | 0.625 |
| Adults, lung opacity vs normal | 4,242 | 0.953 | 0.781 | 0.960 | 0.887 | 0.593 |
| Adults, lung opacity vs everything else | 7,694 | 0.824 | 0.781 | 0.737 | **0.747** | **0.776** |

Simple baselines, AUROC in the same three groups: raw 32x32 pixels + logistic regression 0.847 / 0.835 / 0.727;
frozen ImageNet ResNet-18 + logistic regression 0.953 / 0.876 / 0.752. The fine-tuned model beats both.

What goes wrong on adults:

- Realistic mix: accuracy 74.7% is below the 77.6% of always answering "no pneumonia".
- It flags 42.4% of adults who are sick with something else (normal: 3.9%, lung opacity: 78.1%).
- One threshold does not fit both views. Standing (PA): sensitivity 0.277, specificity 0.949. Bedside (AP): sensitivity 0.930, specificity 0.362. AUROC inside each view (vs everything else): PA 0.795, AP 0.727.
- Scores are miscalibrated: ECE 0.136 on children, 0.397 on adults.
- By age (vs everything else): 18 to 39 0.841, 40 to 59 0.826, 60+ 0.802. Women 0.818, men 0.829.

The "unlike my training data" warning (Mahalanobis distance on the 512 embedding numbers, limit = 95th percentile of children's validation data):

- Fires for 7.2% of the children's test X-rays and 80.6% of adults (AUROC children vs adults 0.957).
- It detects the hospital, not the age: it also fires for 97.1% of the other hospital's patients under 18 (n=311).
- It does not predict which adult answers are wrong (AUROC 0.307). It is a "wrong population" alarm, not an error detector.

Repair with adult X-rays, last layer only (AUROC adults vs everything else): 0 → 0.821, 100 → 0.781, 1,000 → 0.806, 15,142 → 0.839.
Retraining only the last layer barely helps. Whole-network results come from step 07.

Audit (`results/audit.json`):

- No image is shared between train and test or between the datasets. 32 exact duplicate copies inside single Kermany splits.
- The Kaggle RSNA `patientId` is an exam id. A random split puts the same person in train and test (5,493 of 8,006 test exams in our first attempt). We split by the real NIH patient number instead. Worth telling: we found and fixed our own leak.
- Shortcuts in the children's data: image size alone predicts the label with AUROC 0.879, brightness alone 0.871.
- Shortcut in the adult data: the view alone (AP or PA) gives AUROC 0.797 (vs normal) and 0.706 (vs everything else). Part of the model's adult score may come from "bedside X-ray = sick patient". Age and sex alone: 0.50.
- A simple classifier tells the two hospitals apart with AUROC 0.998, and 0.951 for children of the same age.
- Validation overstates test on the children's data (even raw pixels reach 0.99 on validation and 0.847 on test), so the official test set differs from the training pool.

## 5. The story (be honest, this is where the points are)

The dramatic guess "the model collapses on adults" is **wrong**, and three papers in 2026 already tested this dataset pair
(`docs/RESEARCH_NOTES.md` section 3.2). Present the drop as a replication and cite them. Our own contribution:

1. The headline number hides the failure: ranking looks fine (0.82 to 0.95) while the tool as used is worse than a constant answer.
2. The failure has a shape: the threshold breaks by X-ray view, and "sick with something else" is mistaken for pneumonia.
3. A built-in warning that notices the wrong population, tested, including what it cannot do.
4. How many local X-rays a clinic needs to repair the model (last layer vs whole network).
5. A heatmap test against radiologists' boxes with no-skill reference points, not just pretty pictures.
6. A leak in the usual way of splitting the adult dataset, found and fixed.

Phrase novelty as "to our knowledge". Never change the threshold rule after seeing test results.

## 6. To do, most important first

1. Check section 3: what arrived, look at every figure, read the outputs of steps 05 (second model), 06 and 07.
2. **`README.md` as the write-up.** One page first: problem and user, approach, results table with baseline, what did not work, limits. Then details, how to run, and all citations (ready to paste in `docs/RESEARCH_NOTES.md` section 8, impact numbers in section 4). Plain English, no jargon without explanation.
3. **A demo link that judges can open.** Options: (a) the team buys Hugging Face PRO, then `python scripts/deploy_space.py space RL8596/pneumoshift`; (b) a last notebook cell that starts `app/app.py` with a public Gradio share link (good enough for the video and the live final); (c) a static Space that runs the model in the browser with ONNX Runtime Web (free and permanent, most work).
4. Run the notebook in Colab once, in quick mode, and fix what breaks. Ask the team to do the run if you cannot.
5. `docs/VIDEO_SCRIPT.md`: problem 15 s, demo 60 s, how it works 30 s, limits and next steps 15 s. The demo moment: the same adult X-ray, the children-only model is confidently wrong, the warning fires, the second model disagrees.
6. App polish: model cards and the limits panel should state the numbers above. Explain that the score is not a calibrated probability.
7. Ask the team whether to merge this branch into `main` (judges see `main`) and remind them to make the repo public.
8. Update `docs/PLAN.md`.

## 7. Ideas that would add points (pick by time left)

- **Local recalibration curve** (no training, minutes): with n labelled local X-rays, re-choose only the threshold, per view. How many does a clinic need to get back to 95% sensitivity? Very practical for our user and, to our knowledge, not in the papers.
- **What are the false alarms?** Join the adult exams to the NIH ChestX-ray14 finding labels through `data_cache/raw/rsna/rsna_nih_mapping.json` and report which diseases the model mistakes for pneumonia.
- **Reliability diagram** children vs adults for the calibration claim.
- **Does the second model fix the view problem?** Compare PA and AP sensitivity of both models at their own thresholds.
- **Three training seeds** for the children-only model on a GPU, to show the result is stable.
- **Impact framing** from `docs/RESEARCH_NOTES.md` section 4: more people aged 70+ than children under 5 die of lower respiratory infections, yet the tutorial dataset has only children.

## 8. Pitfalls

- `artifacts/` and `data_cache/` are not in git. `results/` is.
- Do not use the Hugging Face "Epic Chittagong" dataset as independent test data: it contains copies of the Kermany images.
- The adult dataset contains minors (1,276 X-rays under 18). `is_adult` filters them; they are useful for the "same age, other hospital" check, but the groups are small (21 test X-rays aged 1 to 5).
- The app pins Gradio 6.32.0 and Python 3.13 (`app/README.md`, `app/requirements.txt`).
- The demo must keep its "research only, not a medical device" notice.

## 9. What only the two humans can do

- Record the demo video (2 minutes at most), upload it unlisted to YouTube or to Google Drive with link sharing.
- Open every link in a logged-out browser window.
- Fill in the submission form with both names.
- Read the write-up until both can explain every number. The final round is a live demo with questions.
