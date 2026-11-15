# Protocol for the second generation of models ("v2")

Written on Sat 14 Nov 2026, about 16:00 UTC, and committed to git **before any v2 model was trained**.
The git history shows the order. Nothing in this file may be changed after the first v2 run, except to add a dated note.

## Why this file exists

The first two models ("v1": `kids_only` and `kids_plus_adults`) are published and tested. We now train more
models on a GPU. When many models are trained, it is easy to pick the one with the best test numbers and to
choose the thresholds that look best. That would make the test numbers meaningless. So everything that could be
chosen later is fixed here first.

## What we already knew when this was written

All test results of the two v1 models, including the errors by X-ray view and by diagnosis class
(`results/shift_*.json`). Those results shaped this plan: the use of all adult training X-rays, the threshold
on adult validation data, and the second operating point. So the adult test set is no longer untouched for
*designing* v2. Only the one primary comparison below counts as a confirmation. Everything else is description.

## One recipe for every v2 run

ResNet-18 pretrained on ImageNet (`timm/resnet18.tv_in1k`), 224 px input, 6 epochs, learning rate 3e-4, batch 32,
the same augmentation as v1, no mixed precision. **The last epoch is kept** (v1 kept the best epoch on validation
data). Only two things differ between runs: which X-rays are used for training, and the random seed. So a
difference between two runs can be put down to the data or to chance, and the seeds show how large chance is.

The exact list of runs is `python scripts/09_gpu_batch.py list` (22 runs, four stages, most important first).

Not done, and why:

- **Chest-X-ray-pretrained weights.** Our adult X-rays come from the NIH ChestX-ray14 collection. Networks
  pretrained on it have already seen our test X-rays.
- **Higher resolution, other networks as candidates, longer training, parameter sweeps, radiologist boxes as an
  extra training signal.** Each adds a choice that would have to be made on a validation set of only 576 adult
  cases (from 221 patients), where differences below about 0.01 AUROC are noise. One ResNet-50 run is included
  as a side note and can never become the headline model.

## The headline model is fixed by name

**`v2_kids_plus_all`**: children's training X-rays + all 15,142 adult training X-rays, seed 2026.
No other run can take its place because of its results. If the PC delivers nothing by Sat 21:00 UTC, the same
recipe is trained on the cloud CPU under the name `v2cpu_kids_plus_all` and takes its place (decided by arrival
time, not by numbers).

## What "a good repaired model" means

Fixed now, so that it cannot be bent to fit the result.

1. **Primary comparison (the only confirmation).** AUROC on the adult test X-rays (lung opacity vs everything
   else, n = 7,694), `v2_kids_plus_all` against the published `kids_plus_adults`. We report the difference with a
   95% interval from resampling patients (not single X-rays: one patient can have several). The repair is
   called better only if that interval lies above zero.
2. **Description, reported for every model:**
   - AUROC inside standing X-rays (PA) and inside bedside X-rays (AP). A model could raise its overall AUROC just
     by learning "bedside X-ray = sick patient". It is only called better if it is better inside both views.
   - 15,142 adult X-rays against 6,000 and against 0, same recipe, three seeds each (range of three runs).
   - Baselines **trained on the same adult X-rays**: (a) view, age and sex only, no picture;
     (b) a frozen ImageNet network plus logistic regression; (c) "flag every bedside X-ray"; (d) "always no".
   - The adults-only model, to answer one question: do the children's X-rays help or hurt on adults?
   - How well the score tells bedside from standing X-rays among Normal adults (the shortcut check).
   - Calibration error on adults and on children. Results on children at the model's own threshold.
3. **Not the definition of good:** beating "always no pneumonia" in accuracy at the screening threshold. Only
   22.4% of the adults have lung opacity, so "always no" is right 77.6% of the time while catching nobody. At 95%
   sensitivity a model beats that only with a specificity above 72.6%, which needs an AUROC of about 0.93 to 0.94
   (estimate). We expect about 0.88 to 0.90. We say so in advance.

## Operating points

Every threshold is chosen on **validation** X-rays and then applied unchanged to the test X-rays. One script
computes all of them from the stored scores, for every model including the two v1 models.

| Point | Rule | Status |
|---|---|---|
| A. Screening | highest threshold that still catches 95% of the lung-opacity cases among **adult** validation X-rays | the project's rule, applied to the population the clinic serves |
| B. As first implemented | the same 95% rule on children's + adult validation X-rays pooled | the rule as v1 `kids_plus_adults` used it; reported for every model so the change is visible |
| C. High-confidence | lowest threshold at which 95% of adult validation X-rays **without** lung opacity are not flagged | **added on 14 Nov, after the v1 test results were seen**; the mirror image of A, no free number |

Disclosure for A: v1 `kids_plus_adults` chose its threshold on the pooled set. The children's cases are easy, so
almost all permitted misses fell on adults, and adults got 91% sensitivity instead of 95%. Applying the rule to
adult validation data is a clarification made after seeing that result. Therefore B is always shown next to A.

About C: accuracy beats "always no" whenever sensitivity divided by (1 minus specificity) is above 3.46, which
almost any strict threshold achieves, even for the children-only model. So at point C we report sensitivity, and
we do not present "more accurate than always no" as proof of a good model.

In addition we read two summaries off each test curve: specificity at exactly 95% sensitivity and sensitivity at
exactly 95% specificity. They describe the curve. They are not thresholds anyone could have set in advance.

Intervals for sensitivity and specificity resample validation patients and test patients together, so they
include the uncertainty of the threshold.

## Tables fixed in advance

- **Main adult table**, at most eight rows: always no, flag every bedside X-ray, the adult-trained baselines,
  `kids_only`, `kids_plus_adults`, `v2_kids_plus_6000`, `v2_kids_plus_all`, `v2_adults_all`.
- **Standing X-rays only (PA).** Our user is a small clinic, which takes standing X-rays. Among those adults only
  9.3% have lung opacity and "always no" is right 90.7% of the time. This table is the one closest to the user.
  It rests on 393 test cases, so its intervals are wide and are shown.
- **Repair curve:** adult AUROC against the number of adult training X-rays (0, 300, 1,000, 3,000, 6,000, 15,142),
  whole network, next to the last-layer-only curve from v1. v1 models appear as separate labelled points.

## Test-data discipline

- The training machine trains with `--no-test`. It stores validation results and each model's scores for all
  X-rays. It never computes or shows a test result, and never runs the analysis steps.
- Test results of v2 models are computed once, in the cloud session, by one script, after the validation
  summary of the finished runs is committed.
- Nothing new is trained after Sun 15 Nov 10:00 UTC. No new number enters the write-up after 12:00 UTC.

## Caveats that every result sentence must carry

- "Lung opacity" is a radiologist's finding on the picture, mostly from one reader, not a confirmed pneumonia.
- With three seeds we report the range of three runs, not a confidence interval.
- For a model trained on children and adults, the "unlike my training data" warning means "unlike this model's
  own training mix". It fires mostly on children (v1: 24% of children, 1% of adults).
- No in-hospital ceiling is known: nobody in our sources reports a model trained and tested on these adults.

## Dated notes (added after training had started; nothing above was changed)

**Sat 14 Nov 2026, 16:45 UTC.** Written when 3 of 22 runs had arrived, after an independent review of the code
and of this file. No test result of any v2 model had been computed.

1. *The fallback was not used.* The PC delivered `v2_kids_plus_all` at 16:10 UTC.
2. *What "no test result on the training machine" means exactly.* The training machine computes no test metric
   of any v2 model. Its preflight recomputes one published v1 number (adult AUROC 0.874) to prove that labels and
   split are identical there. Every run stores its scores for all X-rays, test X-rays included; metrics on them
   are computed only by `scripts/10_v2_report.py`, in the cloud session.
3. *Three things differ between runs, not two.* Besides the training X-rays and the seed: the children-only runs
   (`v2_kids_plus_0`, and its two other seeds) fit their temperature and their stored threshold on children's
   validation X-rays, all other runs on adult validation X-rays. The kept weights are not affected. The report
   therefore gives results at thresholds A and B, which it computes the same way for every model, and gives the
   calibration error both with each model's own temperature and with one fitted on adult validation X-rays.
4. *The seed also decides which adult X-rays are drawn.* At 300 to 6,000 adult X-rays each seed draws its own
   X-rays, and the draws are not subsets of one another. The range over seeds covers the draw and the training.
   `v2_kids_plus_6000` (seed 2026) was trained on exactly the 6,000 adult X-rays of the published
   `kids_plus_adults`, so those two differ in the recipe only (last epoch, validation on adults, GPU).
5. *The main table has nine rows, not eight.* The list in "Tables fixed in advance" has nine entries;
   "at most eight" was a miscount. No row was added or dropped after results.
6. *The test results are computed once, after the last run has arrived.* `scripts/10_v2_report.py test` refuses to
   run a second time without a written reason, and refuses runs that do not follow the recipe above.
7. *An exploratory check was added* after the first v2 validation result showed that 15,142 adult X-rays give the
   same adult validation score as 6,000 (0.874 and 0.875): a public network pretrained on chest X-rays of another
   hospital (`scripts/11_xray_pretrained_check.py`). Its rows are labelled exploratory wherever they appear and are
   not part of the comparison above.

**Sun 15 Nov 2026** (the exact time is the commit time of this note). Nothing above was changed.

8. *A second exploratory check was added on Sun 15 Nov, 11:45 UTC,* after all test results above were known: the
   average of the three `v2_kids_plus_all` runs (`docs/V2_ADDENDUM_ENSEMBLE.md`, `scripts/13_ensemble_check.py`).
   Nothing was trained. It used the test X-rays a second time. Its numbers entered the write-up before the 12:00
   UTC limit above; later edits changed wording only. It is labelled exploratory wherever it appears and does not
   change the primary comparison or its verdict.
