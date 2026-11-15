# Averaging the networks we already trained (written by scripts/13_ensemble_check.py)

**Exploratory.** Rules: docs/V2_ADDENDUM_ENSEMBLE.md, committed before any of these numbers was computed. Added after the test
results of the single models were seen, so this does not replace the verdict of the protocol's primary comparison.
No training: the scores of finished runs are averaged. 'Lung opacity' is a radiologist's finding, not a confirmed pneumonia.

## Adults, lung opacity vs everything else (n = 7,694, 22.4% with lung opacity)

| Model | Networks | AUROC | inside standing (PA) | inside bedside (AP) | A: sens / spec | A: right answers | C: sens / spec |
|---|---|---|---|---|---|---|---|
| kids_plus_adults | 1 | 0.874 (0.861 to 0.886) | 0.865 | 0.811 | 0.96 / 0.50 | 0.605 | 0.46 / 0.94 |
| v2_kids_plus_all | 1 | 0.880 (0.867 to 0.892) | 0.876 | 0.823 | 0.96 / 0.51 | 0.609 | 0.49 / 0.95 |
| ensemble_3 | 3 | 0.886 (0.874 to 0.898) | 0.888 | 0.828 | 0.97 / 0.54 | 0.638 | 0.49 / 0.95 |
| ensemble_7 | 7 | 0.889 (0.877 to 0.901) | 0.892 | 0.831 | 0.96 / 0.56 | 0.652 | 0.50 / 0.94 |

## Paired differences in adult AUROC (same patient resamples as results/v2_report.md)

| Comparison | Adults | inside PA | inside AP |
|---|---|---|---|
| ensemble_3 minus kids_plus_adults | +0.012 (+0.007 to +0.017) | +0.023 (+0.012 to +0.034) | +0.017 (+0.007 to +0.026) |
| ensemble_3 minus v2_kids_plus_all | +0.006 (+0.004 to +0.008) | +0.012 (+0.007 to +0.017) | +0.005 (+0.001 to +0.008) |
| ensemble_7 minus kids_plus_adults | +0.015 (+0.010 to +0.020) | +0.027 (+0.017 to +0.038) | +0.020 (+0.010 to +0.029) |
| ensemble_7 minus v2_kids_plus_all | +0.009 (+0.007 to +0.012) | +0.016 (+0.010 to +0.023) | +0.008 (+0.004 to +0.012) |
| ensemble_7 minus ensemble_3 | +0.003 (+0.002 to +0.004) | +0.004 (+0.001 to +0.007) | +0.003 (+0.001 to +0.005) |
| v2_kids_plus_all minus kids_plus_adults | +0.006 (-0.000 to +0.011) | +0.011 (-0.000 to +0.023) | +0.012 (+0.001 to +0.022) |

Rule of the protocol applied to ensemble_3 minus kids_plus_adults: called better only if the overall interval lies above zero and the difference is positive inside both views. **Result: the averaged model is better (exploratory).**

## Points A and C with intervals, standing X-rays, children, further checks

| Model | A: sensitivity | A: specificity | C: sensitivity | C: specificity | PA at A: sens / spec | Children: AUROC | Children at A: sens / spec | Calibration error, adults | Bedside vs standing among Normal adults |
|---|---|---|---|---|---|---|---|---|---|
| kids_plus_adults | 0.96 (0.93 to 0.98) | 0.50 (0.43 to 0.59) | 0.46 (0.39 to 0.54) | 0.94 (0.92 to 0.96) | 0.85 / 0.70 | 0.962 | 1.00 / 0.35 | 0.029 | 0.87 |
| v2_kids_plus_all | 0.96 (0.93 to 0.98) | 0.51 (0.41 to 0.60) | 0.49 (0.41 to 0.55) | 0.95 (0.93 to 0.96) | 0.90 / 0.65 | 0.973 | 1.00 / 0.32 | 0.018 | 0.64 |
| ensemble_3 | 0.97 (0.92 to 0.98) | 0.54 (0.44 to 0.65) | 0.49 (0.43 to 0.57) | 0.95 (0.93 to 0.96) | 0.91 / 0.69 | 0.976 | 1.00 / 0.35 | 0.024 | 0.64 |
| ensemble_7 | 0.96 (0.93 to 0.98) | 0.56 (0.47 to 0.66) | 0.50 (0.44 to 0.58) | 0.94 (0.93 to 0.96) | 0.90 / 0.71 | 0.967 | 1.00 / 0.33 | 0.022 | 0.64 |
