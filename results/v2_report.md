# Model comparison (written by scripts/10_v2_report.py)

Rules: docs/V2_PROTOCOL.md. Every threshold comes from validation X-rays and is applied unchanged to test X-rays.
Intervals are 95% intervals from resampling patients. 'Lung opacity' is a radiologist's finding, not a confirmed pneumonia.
Operating points: A = 95% sensitivity on adult validation X-rays; B = 95% sensitivity on children's + adult validation X-rays
pooled (shown without interval); C = 95% specificity on adult validation X-rays (added 14 Nov, after the v1 test results).
Rows marked exploratory were added after the first v2 validation result and are not part of the protocol.

## Adults, lung opacity vs everything else (n = 7,694 X-rays of 3,307 patients, 22.4% with lung opacity)

| Model | AUROC | inside standing (PA) | inside bedside (AP) | A: sens / spec | B: sens / spec | C: sens / spec |
|---|---|---|---|---|---|---|
| always_no | 0.500 (0.500 to 0.500) | 0.500 | 0.500 | 0.00 / 1.00 | n/a | 0.00 / 1.00 |
| flag_bedside | 0.706 (0.690 to 0.722) | 0.500 | 0.500 | 0.77 / 0.64 | n/a | 0.77 / 0.64 |
| metadata_only | 0.706 (0.683 to 0.730) | 0.496 | 0.502 | 0.96 / 0.12 | n/a | 0.07 / 0.96 |
| frozen_imagenet | 0.848 (0.834 to 0.863) | 0.819 | 0.776 | 0.97 / 0.41 | n/a | 0.39 / 0.94 |
| kids_only | 0.824 (0.809 to 0.839) | 0.795 | 0.727 | 0.95 / 0.44 | 0.92 / 0.51 | 0.29 / 0.94 |
| kids_plus_adults | 0.874 (0.861 to 0.886) | 0.865 | 0.811 | 0.96 / 0.50 | 0.91 / 0.65 | 0.46 / 0.94 |
| v2_kids_plus_6000 | 0.872 (0.859 to 0.885) | 0.874 | 0.805 | 0.96 / 0.52 | 0.92 / 0.63 | 0.48 / 0.94 |
| v2_kids_plus_all | 0.880 (0.867 to 0.892) | 0.876 | 0.823 | 0.96 / 0.51 | 0.92 / 0.64 | 0.49 / 0.95 |
| v2_adults_all | 0.878 (0.865 to 0.890) | 0.876 | 0.818 | 0.95 / 0.56 | 0.92 / 0.65 | 0.49 / 0.94 |
| xray_pretrained_as_is (exploratory) | 0.801 (0.784 to 0.817) | 0.769 | 0.712 | 0.98 / 0.35 | n/a | 0.23 / 0.95 |
| xray_pretrained_frozen (exploratory) | 0.848 (0.833 to 0.862) | 0.830 | 0.768 | 0.96 / 0.45 | n/a | 0.34 / 0.95 |

## Standing X-rays only (PA): n = 4,214, 9.3% with lung opacity ('always no' is right for 90.7%)

| Model | AUROC | A: sensitivity | A: specificity | C: sensitivity | C: specificity |
|---|---|---|---|---|---|
| always_no | 0.500 (0.500 to 0.500) | 0.00 (0.00 to 0.00) | 1.00 (1.00 to 1.00) | 0.00 (0.00 to 0.00) | 1.00 (1.00 to 1.00) |
| flag_bedside | 0.500 (0.500 to 0.500) | 0.00 (0.00 to 0.00) | 1.00 (1.00 to 1.00) | 0.00 (0.00 to 0.00) | 1.00 (1.00 to 1.00) |
| metadata_only | 0.496 (0.459 to 0.534) | 0.82 (0.67 to 0.90) | 0.19 (0.11 to 0.30) | 0.00 (0.00 to 0.00) | 1.00 (1.00 to 1.00) |
| frozen_imagenet | 0.819 (0.797 to 0.840) | 0.88 (0.78 to 0.93) | 0.58 (0.46 to 0.69) | 0.07 (0.04 to 0.12) | 1.00 (0.99 to 1.00) |
| kids_only | 0.795 (0.771 to 0.818) | 0.78 (0.71 to 0.90) | 0.66 (0.52 to 0.72) | 0.01 (0.00 to 0.02) | 1.00 (0.99 to 1.00) |
| kids_plus_adults | 0.865 (0.845 to 0.883) | 0.85 (0.76 to 0.91) | 0.70 (0.62 to 0.80) | 0.16 (0.10 to 0.21) | 0.99 (0.99 to 1.00) |
| v2_kids_plus_6000 | 0.874 (0.857 to 0.890) | 0.90 (0.82 to 0.95) | 0.67 (0.54 to 0.76) | 0.22 (0.15 to 0.33) | 0.98 (0.97 to 0.99) |
| v2_kids_plus_all | 0.876 (0.858 to 0.893) | 0.90 (0.84 to 0.96) | 0.65 (0.54 to 0.74) | 0.25 (0.18 to 0.32) | 0.98 (0.97 to 0.99) |
| v2_adults_all | 0.876 (0.857 to 0.892) | 0.87 (0.81 to 0.95) | 0.70 (0.54 to 0.77) | 0.30 (0.22 to 0.38) | 0.98 (0.97 to 0.98) |
| xray_pretrained_as_is (exploratory) | 0.769 (0.744 to 0.791) | 0.92 (0.86 to 0.96) | 0.46 (0.39 to 0.56) | 0.04 (0.02 to 0.09) | 0.99 (0.98 to 0.99) |
| xray_pretrained_frozen (exploratory) | 0.830 (0.807 to 0.850) | 0.87 (0.80 to 0.93) | 0.61 (0.52 to 0.70) | 0.03 (0.01 to 0.05) | 1.00 (1.00 to 1.00) |

## Adults at point A, with intervals, and further checks

'Always no' is right for 77.6% of these adults. Curve read-offs describe the test curve; they are not thresholds set in advance.

| Model | A: sensitivity | A: specificity | A: right answers | Read off the curve: spec. at 95% sens. | sens. at 95% spec. | Bedside vs standing among Normal adults (0.5 = the score ignores the view) |
|---|---|---|---|---|---|---|
| always_no | 0.00 (0.00 to 0.00) | 1.00 (1.00 to 1.00) | 0.776 | n/a | n/a | 0.50 |
| flag_bedside | 0.77 (0.74 to 0.80) | 0.64 (0.62 to 0.66) | 0.669 | n/a | n/a | 1.00 |
| metadata_only | 0.96 (0.92 to 0.98) | 0.12 (0.07 to 0.19) | 0.310 | 0.14 | 0.09 | 1.00 |
| frozen_imagenet | 0.97 (0.94 to 0.98) | 0.41 (0.32 to 0.50) | 0.533 | 0.48 | 0.37 | 0.81 |
| kids_only | 0.95 (0.93 to 0.98) | 0.44 (0.34 to 0.49) | 0.553 | 0.44 | 0.26 | 0.92 |
| kids_plus_adults | 0.96 (0.93 to 0.98) | 0.50 (0.43 to 0.59) | 0.605 | 0.55 | 0.45 | 0.87 |
| v2_kids_plus_6000 | 0.96 (0.93 to 0.98) | 0.52 (0.40 to 0.61) | 0.616 | 0.56 | 0.44 | 0.69 |
| v2_kids_plus_all | 0.96 (0.93 to 0.98) | 0.51 (0.41 to 0.60) | 0.609 | 0.56 | 0.46 | 0.64 |
| v2_adults_all | 0.95 (0.92 to 0.98) | 0.56 (0.41 to 0.63) | 0.645 | 0.55 | 0.46 | 0.60 |
| xray_pretrained_as_is (exploratory) | 0.98 (0.95 to 0.99) | 0.35 (0.28 to 0.42) | 0.488 | 0.44 | 0.24 | 0.63 |
| xray_pretrained_frozen (exploratory) | 0.96 (0.94 to 0.98) | 0.45 (0.37 to 0.53) | 0.563 | 0.51 | 0.35 | 0.74 |

## Children, and calibration

'Own threshold' is the one stored with the model; the validation X-rays it was chosen on differ between models (last column).

| Model | Children: AUROC | at own threshold: sens / spec | at point A: sens / spec | Calibration error, adults (own temperature) | adults (temperature from adult validation) | children | Own threshold and temperature chosen on |
|---|---|---|---|---|---|---|---|
| kids_only | 0.975 | 0.98 / 0.87 | 1.00 / 0.62 | 0.398 | 0.327 | 0.136 | children val |
| kids_plus_adults | 0.962 | 1.00 / 0.46 | 1.00 / 0.35 | 0.029 | 0.031 | 0.152 | children val + adult val |
| v2_kids_plus_6000 | 0.981 | 1.00 / 0.36 | 1.00 / 0.36 | 0.030 | 0.030 | 0.130 | adult val |
| v2_kids_plus_all | 0.973 | 1.00 / 0.32 | 1.00 / 0.32 | 0.018 | 0.018 | 0.134 | adult val |
| v2_adults_all | 0.930 | 1.00 / 0.36 | 1.00 / 0.36 | 0.023 | 0.023 | 0.266 | adult val |

## Primary comparison (the only confirmation in the protocol)

adult test AUROC (lung opacity vs everything else), v2_kids_plus_all minus published kids_plus_adults, patients resampled 1000 times: **+0.006** (95% interval -0.000 to +0.011). Inside standing X-rays +0.011 (-0.000 to +0.023), inside bedside X-rays +0.012 (+0.001 to +0.022). Rule: called better only if the overall interval lies above zero and the difference is positive inside both views. **Verdict: no improvement is shown.**

## Other paired differences in adult AUROC (description, not confirmation)

| Comparison | Adults | inside PA | inside AP |
|---|---|---|---|
| v2_kids_plus_all minus kids_plus_adults | +0.006 (-0.000 to +0.011) | +0.011 (-0.000 to +0.023) | +0.012 (+0.001 to +0.022) |
| v2_kids_plus_all minus v2_kids_plus_6000 | +0.008 (+0.002 to +0.013) | +0.002 (-0.008 to +0.013) | +0.017 (+0.008 to +0.027) |
| v2_kids_plus_all minus v2_adults_all | +0.002 (-0.002 to +0.006) | +0.000 (-0.009 to +0.010) | +0.004 (-0.002 to +0.011) |
| v2_kids_plus_all minus frozen_imagenet | +0.031 (+0.023 to +0.040) | +0.057 (+0.040 to +0.073) | +0.047 (+0.031 to +0.062) |
| kids_plus_adults minus kids_only | +0.050 (+0.042 to +0.058) | +0.069 (+0.056 to +0.085) | +0.084 (+0.068 to +0.101) |
| kids_plus_adults minus frozen_imagenet | +0.025 (+0.019 to +0.033) | +0.045 (+0.031 to +0.060) | +0.035 (+0.023 to +0.049) |
| v2_kids_plus_6000 minus kids_plus_adults | -0.002 (-0.007 to +0.003) | +0.009 (-0.002 to +0.019) | -0.006 (-0.014 to +0.003) |
| xray_pretrained_frozen minus kids_plus_adults | -0.026 (-0.033 to -0.020) | -0.035 (-0.049 to -0.021) | -0.043 (-0.057 to -0.029) |
| xray_pretrained_frozen minus frozen_imagenet | -0.001 (-0.008 to +0.006) | +0.011 (-0.009 to +0.028) | -0.008 (-0.021 to +0.005) |
| xray_pretrained_as_is minus kids_only | -0.023 (-0.035 to -0.013) | -0.026 (-0.047 to -0.006) | -0.015 (-0.037 to +0.006) |

## Repair curve and seeds (mean and range of the runs, not a confidence interval)

At 300 to 6,000 adult X-rays every seed draws its own X-rays, so the range covers the draw and the training.

| Recipe | adult training X-rays | runs | adult AUROC | standing (PA) | bedside (AP) | children |
|---|---|---|---|---|---|---|
| v2_kids_plus_0 | 0 | 3 | 0.831 (0.820 to 0.841) | 0.802 (0.785 to 0.816) | 0.739 (0.721 to 0.756) | 0.982 (0.979 to 0.983) |
| v2_kids_plus_300 | 300 | 3 | 0.834 (0.829 to 0.837) | 0.795 (0.773 to 0.808) | 0.752 (0.748 to 0.755) | 0.981 (0.977 to 0.985) |
| v2_kids_plus_1000 | 1,000 | 3 | 0.853 (0.850 to 0.856) | 0.830 (0.826 to 0.836) | 0.785 (0.780 to 0.789) | 0.975 (0.971 to 0.979) |
| v2_kids_plus_3000 | 3,000 | 3 | 0.857 (0.846 to 0.863) | 0.841 (0.837 to 0.850) | 0.799 (0.793 to 0.803) | 0.976 (0.970 to 0.981) |
| v2_kids_plus_6000 | 6,000 | 3 | 0.871 (0.870 to 0.872) | 0.867 (0.863 to 0.874) | 0.808 (0.805 to 0.812) | 0.975 (0.965 to 0.981) |
| v2_kids_plus_all | 15,142 | 3 | 0.881 (0.880 to 0.882) | 0.881 (0.876 to 0.886) | 0.822 (0.820 to 0.823) | 0.973 (0.969 to 0.977) |
| v2_kids_plus_all_r50 | 15,142 | 1 | 0.884 (0.884 to 0.884) | 0.882 (0.882 to 0.882) | 0.823 (0.823 to 0.823) | 0.973 (0.973 to 0.973) |
| v2_adults_all | 15,142 | 3 | 0.880 (0.878 to 0.883) | 0.878 (0.876 to 0.884) | 0.822 (0.818 to 0.824) | 0.905 (0.879 to 0.930) |
| kids_only (published v1, best epoch kept) | 0 | 1 | 0.824 | 0.795 | 0.727 | 0.975 |
| kids_plus_adults (published v1, 6,000 adults, best epoch kept) | 6,000 | 1 | 0.874 | 0.865 | 0.811 | 0.962 |
