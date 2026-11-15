# PneumoShift: notes for a Claude session

Read this file, then `docs/HACKATHON_CONTEXT.md` (rules, judging). `docs/RESEARCH_NOTES.md` holds the sources.

**Deadline: Sun 15 Nov 2026, 11:59 PM ET (= Mon 16 Nov, 04:59 UTC).** Round 1 is judged blind from the repo, the
demo link and a 2-minute video. A live final with questions may follow.

## State on Sun 15 Nov 2026, 21:30 UTC

- **The project is finished.** Nothing is running. Do not train anything, and put no new number into the write-up
  (`docs/V2_PROTOCOL.md`: no training after 15 Nov 10:00 UTC, no new number after 12:00 UTC). What remains is
  wording, checks of existing numbers, and the team's own tasks (section 9).
- **The verdict of the planned comparison:** the headline model `v2_kids_plus_all` scores 0.880 on adults, the
  published `kids_plus_adults` 0.874. Difference +0.006 (95% interval -0.000 to +0.011): by the rule fixed in
  advance, no improvement is shown. Do not soften or dramatise it. `scripts/10_v2_report.py test` refuses a second
  run, and that is intended.
- **The extra check** (three networks answering together, 0.886; rules in `docs/V2_ADDENDUM_ENSEMBLE.md`, result in
  `results/exploratory/ensemble.md`) is exploratory. It never replaces the verdict, it is not in the demo, and the
  video script leaves it out. `scripts/13_ensemble_check.py` refuses a second run unless it gets a written reason.
- **`README.md` is the write-up.** Its first part follows the organisers' template; "More results" and "Details"
  hold the rest. On 15 Nov every number was checked against `results/`. Sources 4, 5, 7, 11, 14, 15 and 25 were
  checked in PubMed, and 9, 10, 23 and 24 on the publisher's page or the arXiv listing as found by web search
  (doi.org and arxiv.org cannot be opened from the cloud session). The other sources were not opened again.
  The count of the leak in the first split (5,493 of 8,006) is stored in no result file, because that split was
  not kept. If a judge asks, say "about two thirds".
- **Pictures in the README:** four figures from `results/figures/`, plus a clip and a screenshot of the demo in
  `docs/media/`, made by `node tests/web/record_demo.mjs <folder>` (needs `web/models/`, Playwright and ffmpeg).
  `scripts/06_heatmap_check.py --mistakes-only` redraws the two figures of each model's mistakes without touching
  `results/heatmap_check.json`.
- **Open:** the 2-minute video and its link in the first lines of the README (a comment marks the place). The demo
  is untested on Safari, Firefox and a real phone. The notebook was never run in Colab, and the README says so.
  The title of `results/figures/heatmap_adult_examples.png` says "pneumonia" where "lung opacity" is meant (not
  shown in the README). The first git commit carries a personal e-mail address (blind judging; changing it is the
  team's decision). The weights of the second-generation models exist only on the home PC; the demo does not need
  them.
- Scheduled check-ins (`send_later`, triggers) fail in this setup with "access denied". To be woken later, start a
  background command that ends when the condition is met.

## 0. Rules from the team (follow them)

- The team is two high-school students. They asked for answers in English (14 Nov), short and concrete.
- Usage is limited. Few tool calls, no subagents unless they ask, short answers.
- Goal: win the Veritas AI International High School Hackathon 2026, track **AI + Health**, Advanced level.
- A scam-SMS project is forbidden (it is the organisers' own demo submission).
- Hugging Face must be used. Models are published under the account `RL8596`.
- Never ask for a token in chat. Uploading to `RL8596` needs that account's write token, entered through the secure credential form (host `huggingface.co`). Downloading needs no token.
- No pull request and no push to `main` unless the team asks. Commit and push your work often.
- Judging round 1 is blind: no names or school in the repo. Names go into the submission form only.
- Rule on AI assistance (checked by the team, 15 Nov): AI tools are allowed as long as both team members understand what was built, and they must be listed under Sources. The README does both (Sources and section 11). Do not remove or weaken that disclosure. This file may be deleted before submission if they wish.

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
| Browser demo | `python scripts/08_export_web.py`, then `node tests/web/parity.mjs` | `artifacts/<name>/onnx/`, `web/models/`, `tests/web/reference.json` |

Training one model takes about 1 hour on 4 CPU cores and a few minutes on a GPU. Library code is in `src/pneumoshift/`.
Model: ResNet-18 pretrained on ImageNet (`timm/resnet18.tv_in1k` from the Hub), fine-tuned, one output.
Threshold rule, fixed before testing: 95% sensitivity on validation data. Never tune anything on test data.

## 3. State at handoff

The newest work is on `main`. Work from `main`, and do not merge the older branches
(`claude/loving-hawking-jtiwb3`, `pc/v2-runs`) into it.

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
- **The Gradio Space is not online.** An earlier session reported: upload refused, Gradio Spaces on the free tier need PRO.
  Not verified. Another possible cause: `short_description` in `app/README.md` is longer than the 60 characters the Hub allows
  (the static Space was refused for exactly that until it was shortened).
- **The browser demo is online and tested:** https://huggingface.co/spaces/RL8596/pneumoshift
  (direct: https://rl8596-pneumoshift.static.hf.space/). This is the demo link for the submission.

### Browser demo (`web/`, added Sat 14 Nov, afternoon)

A second version of the demo that needs no server: a static page, both models run in the visitor's browser
with ONNX Runtime Web. It exists because the Gradio Space was refused (see above). Same words and checks as `app/app.py`.

- `scripts/08_export_web.py` exports each model to ONNX (logit, embedding and heatmap in one run), checks it against
  PyTorch, and writes the reference file for the browser test. `push` uploads the files to `onnx/` in each model repo,
  `space RL8596/pneumoshift` uploads `web/` as a static Space. Both uploads need the write token.
- `web/app.js` repeats the Python preprocessing (including Pillow's resize) in JavaScript. `tests/web/parity.mjs`
  opens the page in headless Chromium and compares it with Python on 15 pictures: the model input is identical byte
  for byte, logits differ by less than 0.00001, decisions and warnings are the same. Run it after any change to `web/`.
- The page loads the models from the Hub and falls back to `web/models/` (not in git). `?models=local` skips the Hub.
- Checked on screenshots: desktop, tablet, phone, dark mode. Not checked: Safari and Firefox, a real phone.
- Uploaded on Sat 14 Nov: `onnx/` in both model repos and the static Space `RL8596/pneumoshift`. The live page passes
  `node tests/web/parity.mjs --url https://rl8596-pneumoshift.static.hf.space/`. After changing `web/`, upload again with
  `python scripts/08_export_web.py space RL8596/pneumoshift --live`, but test in a staging Space first (the token is saved on the cloud environment `VeritasAI_Hackathon`;
  set `HF_TOKEN=hf_proxyinjected` for the command, the proxy puts the real token in).
- If `app/examples/` changes, rerun the export script: it copies the examples to `web/examples/`.

- **New look since Sat 14 Nov, evening (first session, at the team's request):** dark "reading room console" design:
  IBM Plex typeface served from `web/fonts/` (SIL Open Font License), corner marks and labels on the pictures, the score
  as a gauge with the cut-off marked, a strip of facts per model read from its config (`specHtml` in `web/app.js`), a
  colour scale under the heatmap. Words, checks and results are unchanged and the browser test passes. It was tested
  on the staging Space `RL8596/pneumoshift-staging` first, then uploaded to the live Space with `--live`.
  The Gradio app in `app/` keeps the old look. Pull `main` before you continue: it now contains this change.

### Second generation of models, "v2" (started Sat 14 Nov, 16:00 UTC)

The team wants a stronger repaired model and has a Windows PC with an RTX 3080 at home. Read `docs/V2_PROTOCOL.md`
first: it was committed before any v2 model was trained and fixes the recipe, the 22 runs, the headline model
(`v2_kids_plus_all`, by name), the primary comparison and the three operating points. Do not change it.

- **The PC is driven by a second Claude session on that machine** (`docs/PC_RUNBOOK.md`). It works on the branch
  `pc/v2-runs` and commits only `results/v2/<name>/` (settings, scores for every X-ray, no weights) and
  `results/val_v2_*.json`. It never computes test results. Follow its progress with
  `git fetch origin pc/v2-runs` and `git show origin/pc/v2-runs:results/v2/STATUS.json`.
  The team must approve anything that pushes or deletes on the PC in that session's own chat.
- `scripts/00_preflight.py` (lock files in `data_splits/`) proved that the PC has the same split and the same
  pictures. `scripts/09_gpu_batch.py` trains the fixed list (`list`, `smoke`, `run --stage N --commit`, `status`,
  `upload`). On the PC: 158 images per second, about 3 hours for all 22 runs.
- **When runs have arrived:** copy the result files from `origin/pc/v2-runs` into this branch, WITHOUT the status
  file and the smoke-test leftover (otherwise the PC's next pull ends in a merge conflict):
  `git checkout origin/pc/v2-runs -- results/v2 ':(exclude)results/v2/STATUS.json' ':(exclude)results/v2/v2_smoke'`
  and the same for `results/val_v2_*.json`. Never push to `pc/v2-runs` from here. Then run
  `python scripts/10_v2_report.py val`, commit, then `python scripts/10_v2_report.py test` (once), then fill
  README section 6 and the video script. `results/v1_rehearsal.*` is the same report for the two v1 models only.
- **Weights of the headline model** reach the Hub only after a person logs in on the PC (`hf auth login`) and the PC
  runs `python scripts/09_gpu_batch.py upload v2_kids_plus_all`. Then here: pull it, check that its scores equal
  the PC's, run 05 for it, and decide with the team whether it replaces the second card of the demo
  (only through a staging Space, never by editing the live one first).
- **Guards:** `02_train.py`, `hub_models.py push`, `08_export_web.py push` and `deploy_space.py` refuse to touch the two
  published models; `deploy_space.py space` and `08_export_web.py space` (without `--live`) refuse the live Space. The Hub repos and the Space carry the tag `v1-before-retraining`.
- New finding from the rehearsal, already in the README: a frozen ImageNet network + logistic regression trained on
  the adult X-rays reaches 0.848 on adults. That is the fair baseline for any adult-trained model.
- Multi-agent reviews are expensive: one plan review used most of a 5-hour usage allowance. Ask before starting one.
- The two flaws of the `06` mistakes figures were fixed on 15 Nov (see "Review" above). Left: "pneumonia" in the
  title of `heatmap_adult_examples.png`. One early commit in the git history carries a personal e-mail address:
  tell the team before the repo is made public (blind judging).

## 4. Results so far (children-only model, held-out test patients)

Threshold 0.985, temperature-scaled scores.

| Test group | n | AUROC | Sensitivity | Specificity | Accuracy | "Always the majority class" |
|---|---|---|---|---|---|---|
| Children 1 to 5, training hospital | 624 | 0.975 | 0.979 | 0.868 | 0.938 | 0.625 |
| Adults, lung opacity vs normal | 4,242 | 0.953 | 0.781 | 0.960 | 0.888 | 0.594 |
| Adults, lung opacity vs everything else | 7,694 | 0.824 | 0.781 | 0.738 | **0.747** | **0.776** |

Simple baselines, AUROC in the same three groups: raw 32x32 pixels + logistic regression 0.847 / 0.835 / 0.727;
frozen ImageNet ResNet-18 + logistic regression 0.953 / 0.876 / 0.752. The fine-tuned model beats both.

What goes wrong on adults:

- Realistic mix: accuracy 74.7% is below the 77.6% of always answering "no pneumonia".
- It flags 42.5% of adults who are sick with something else (normal: 4.0%, lung opacity: 78.1%).
- One threshold does not fit both views. Standing (PA): sensitivity 0.277, specificity 0.949. Bedside (AP): sensitivity 0.930, specificity 0.362. AUROC inside each view (vs everything else): PA 0.795, AP 0.727.
- Scores are miscalibrated: ECE 0.136 on children, 0.398 on adults.
- By age (vs everything else): 18 to 39 0.841, 40 to 59 0.826, 60+ 0.802. Women 0.818, men 0.829.

The "unlike my training data" warning (Mahalanobis distance on the 512 embedding numbers, limit = 95th percentile of children's validation data):

- Fires for 7.2% of the children's test X-rays and 80.6% of adults (AUROC children vs adults 0.957).
- It detects the hospital, not the age: it also fires for 97.1% of the other hospital's patients under 18 (n=311).
- It does not predict which adult answers are wrong (AUROC 0.308). It is a "wrong population" alarm, not an error detector.

Repair with adult X-rays, last layer only (AUROC adults vs everything else): 0 → 0.821, 100 → 0.781, 1,000 → 0.806, 15,142 → 0.839.
Retraining only the last layer barely helps. Whole-network results come from step 07.

Second model, children + 6,000 adult X-rays, whole network retrained (arrived after the lines above were written; threshold 0.157):

| Test group | AUROC | Sensitivity | Specificity | Accuracy | "Always the majority class" |
|---|---|---|---|---|---|
| Children 1 to 5 | 0.962 | 1.000 | 0.457 | 0.796 | 0.625 |
| Adults, lung opacity vs normal | 0.979 | 0.911 | 0.948 | 0.933 | 0.594 |
| Adults, lung opacity vs everything else | 0.874 | 0.911 | 0.650 | **0.708** | **0.776** |

- Adult data helps ranking (0.824 to 0.874) and sensitivity (0.781 to 0.911), but the children pay: their specificity falls from 0.868 to 0.457.
- Still below the constant answer in the realistic mix. Say honestly that a 95%-sensitivity screening threshold trades accuracy for caught cases, so also report sensitivity and specificity, not accuracy alone.
- It flags 56.8% of adults who are sick with something else (normal 5.2%, lung opacity 91.1%). This confusion is the hard part and adult data did not fix it.
- The view problem stays: PA sensitivity 0.715, specificity 0.842; AP sensitivity 0.969, specificity 0.307.

Heatmap pointing test (`results/heatmap_check.json`, 1,500 adult test X-rays with radiologist boxes; hottest Grad-CAM point inside a box):

| Random point | Image centre | Always the most common spot | Children-only model | Children + adults model |
|---|---|---|---|---|
| 15% | 10% | 57% | 25% | 52% |

- The children-only model rarely looks where the radiologists marked. Adult data doubles the hit rate, yet a fixed spot chosen without looking at the image still does better.
- Caveat to state: Grad-CAM on ResNet-18 has a 7x7 grid, so it is coarse. Do not sell the heatmaps as localisation.

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

1. Check section 3: what arrived, look at every figure, read the outputs of steps 05 (second model), 06 and 07. DONE except step 07 (its models come from the v2 runs) and small flaws in the 06 figures.
2. **`README.md` as the write-up.** DRAFT DONE (v2 results still to be added in its section 6; every number should be checked once more against `results/`). One page first: problem and user, approach, results table with baseline, what did not work, limits. Then details, how to run, and all citations (ready to paste in `docs/RESEARCH_NOTES.md` section 8, impact numbers in section 4). Plain English, no jargon without explanation.
3. **A demo link that judges can open.** DONE with option (c), see "Browser demo" in section 3. The options were: (a) the team buys Hugging Face PRO, then `python scripts/deploy_space.py space RL8596/pneumoshift`; (b) the public Gradio share link: already built as notebook section 9 (`PNEUMOSHIFT_SHARE=1 python app/app.py`), never tested, good enough for the video and the live final; (c) a static Space that runs the model in the browser with ONNX Runtime Web (free and permanent, most work).
4. Run the notebook in Colab once, in quick mode, and fix what breaks. Ask the team to do the run if you cannot.
5. `docs/VIDEO_SCRIPT.md`: DRAFT DONE. Problem 15 s, demo 60 s, how it works 30 s, limits and next steps 15 s. The demo moment: the same adult X-ray, the children-only model is confidently wrong, the warning fires, the second model disagrees.
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
