# Addendum to the v2 protocol: averaging the networks we already trained

Written on Sun 15 Nov 2026, 12:00 UTC, **before any number of an averaged model was computed**, on validation
or on test X-rays. This file is committed before `scripts/13_ensemble_check.py` is run.

## Status: exploratory

The test results of the single models were computed on 14 Nov and we have seen them (`results/v2_report.md`).
The test X-rays are therefore no longer untouched. Whatever comes out here is reported as exploratory. It does
not replace the verdict of the protocol's primary comparison ("no improvement is shown"), and it cannot become
the headline result.

## Why this, and not something else

The team asked what would make the model more accurate in the time left (the deadline is 16 Nov, 04:59 UTC).
We looked at the options again:

| Option | What published work says | Cost for us |
|---|---|---|
| Average several trained networks (an "ensemble") | A common last step in chest X-ray work. The CheXpert baseline averages 30 saved networks (Irvin et al. 2019). A mix of different networks gained about 0.012 AUROC over its best member on NIH chest X-rays (SynthEnsemble, 2023). A prize-winning team of the RSNA pneumonia challenge reports 0.231 for its ensemble against 0.217 for one model, on the challenge's own detection score (DeepRadiology team, 2018). | No training. We already have three runs of the headline recipe. Minutes on a CPU. |
| Sharper pictures (more than 224 pixels) | Mixed. One study found the best mean AUROC at 1,024 pixels (0.842 against 0.827 at 256; Wollek et al. 2023). Another found the best size depends on the finding, with gains mainly for small findings, and losses in some cases (Sabottke and Spieler 2020). | New data preparation, new training on the home PC (5 to 6 hours), one more choice to make on a small validation set. The PC is switched off today. |
| A third answer "sick with something else", or the radiologists' boxes as a training signal | Used by challenge teams together with detection networks. | New code and new training. The PC is switched off today. |

Averaging is the only option that needs no new training, cannot break anything that exists, and has no setting
that we could tune afterwards.

## What is fixed

1. **No training and no new network.** Only the stored scores in `results/v2/<name>/logits.npz` are used.
2. **The members are fixed by name. Nothing is selected.**
   - `ensemble_3` (the main one): `v2_kids_plus_all`, `v2_kids_plus_all_s2027`, `v2_kids_plus_all_s2028`. This is
     the headline recipe trained three times with different random seeds.
   - `ensemble_7` (described only): the three above, plus `v2_adults_all`, `v2_adults_all_s2027`,
     `v2_adults_all_s2028` and `v2_kids_plus_all_r50`. These are all seven networks trained with all 15,142
     adult X-rays.
3. **How the average is taken.** Each member's score is turned into a probability with that member's own
   temperature (fitted on adult validation X-rays). The ensemble's score is the plain mean of these
   probabilities. No weights, no second model on top.
4. **Thresholds** come from adult validation X-rays, exactly as in the protocol: point A (95% sensitivity) and
   point C (95% specificity), applied unchanged to the test X-rays.
5. **The comparison.** Adult test AUROC (lung opacity vs everything else), `ensemble_3` minus the published
   `kids_plus_adults`, with the same patient resamples as the report (1,000, same random seed). The same rule
   as in the protocol: called better only if the overall interval lies above zero and the difference is positive
   inside both X-ray views. Also reported: `ensemble_3` minus the single `v2_kids_plus_all`, and the same two
   differences for `ensemble_7`.
6. **Order.** Validation numbers first (`13_ensemble_check.py val`), committed. Then the test numbers once
   (`13_ensemble_check.py test`). The script refuses a second test run.
7. **Reported whatever comes out:** AUROC overall and inside each view, sensitivity and specificity at points A
   and C, the children's test X-rays, the calibration error, the shortcut check (bedside vs standing among
   Normal adults), and the price: an ensemble of three runs three networks, so it is three times slower.

## What does not change

- The two published models, the live demo and `results/v2_report.*` stay as they are.
- The weights of the three members exist only on the home PC. An ensemble in the demo would need them uploaded
  and would triple the download and the computing time in the browser. That is a later decision of the team.

## Sources

Found by web search on 15 Nov 2026. The numbers above are from the abstracts and search summaries; the full
papers could not be opened from this session (arxiv.org is not reachable from it), so check them before quoting.

- Irvin J, Rajpurkar P, Ko M, et al. CheXpert: a large chest radiograph dataset with uncertainty labels and
  expert comparison. AAAI 2019. (The "30 saved networks" is from memory of the paper, not rechecked.)
- SynthEnsemble: a fusion of CNN, vision transformer, and hybrid models for multi-label chest X-ray
  classification. arXiv:2311.07750, 2023. (Best single network 0.842, ensemble 0.854, NIH ChestX-ray14.)
- The DeepRadiology team. Pneumonia detection in chest radiographs. arXiv:1811.08939, 2018.
- Wollek A, et al. Higher chest X-ray resolution improves classification performance. arXiv:2306.06051, 2023.
- Sabottke CF, Spieler BM. The effect of image resolution on deep learning in radiography. Radiology:
  Artificial Intelligence, 2020. https://pmc.ncbi.nlm.nih.gov/articles/PMC8017385

## Dated notes (added after the check; nothing above was changed)

**Sun 15 Nov 2026** (the exact time is the commit time of these notes). Written after an independent review of
this check.

1. *The time in the first line is wrong.* It says 12:00 UTC. Git shows the real times (UTC): rules and script
   committed at 11:45:03 (`cc68199`), validation numbers at 11:45:42 (`98e93bd`), test numbers computed at 11:45
   and committed at 11:47:11 (`d9bd387`). The order stated above is true. The rules were fixed about one minute
   before the numbers were computed, by the same session, and after all single test results were known. That is
   why this check is exploratory.
2. *The plan's time limits were kept.* `docs/V2_PROTOCOL.md` says that nothing new is trained after 15 Nov 10:00
   UTC and that no new number enters the write-up after 12:00 UTC. Nothing was trained here, and the numbers
   entered the write-up at 11:47 UTC. Later edits changed wording only.
3. *Two small differences between this file and the script.* The script also reports `ensemble_7` minus
   `ensemble_3`, which point 5 does not list. Point B of the protocol was not computed for the averaged models,
   only points A and C.
4. *"Refuses a second test run" means:* unless it is given a written reason (`--again "reason"`). That was not
   used: `results/exploratory/ensemble.json` stores `computed_again_because: null`.
5. *The wording of `results/exploratory/ensemble.md` was made clearer.* Its first version ended with "Result: the
   averaged model is better (exploratory)". That could be read as a second verdict. The file was written again
   from the stored numbers (`python scripts/13_ensemble_check.py render`, which computes nothing). No number
   changed; the git history of the file shows it.
6. *What this check changed, and what it did not.* It added this file, the script and three result files. It
   added text to `README.md`, `CLAUDE.md` and `docs/VIDEO_SCRIPT.md`, and one dated note to
   `docs/V2_PROTOCOL.md`. The two published models, the live demo and all earlier results are untouched (checked
   on 15 Nov against git and against the tag `v1-before-retraining` on the Hub). No model was created or
   uploaded: an averaged model exists only as these tables.
7. *How to undo it.* List the commits of this check with
   `git log --oneline -- docs/V2_ADDENDUM_ENSEMBLE.md results/exploratory/ensemble.json results/exploratory/ensemble_validation.json results/exploratory/ensemble.md`
   and pass them to `git revert`, newest first. That removes the check from the repository and the write-up and
   touches nothing else. A revert adds commits and deletes no history, so it can itself be undone.
