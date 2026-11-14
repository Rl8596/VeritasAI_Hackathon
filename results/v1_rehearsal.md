# Model comparison (written by scripts/10_v2_report.py)

Rules: docs/V2_PROTOCOL.md. Every threshold comes from validation X-rays. Intervals: 95%, patients resampled.

## Adults, lung opacity vs everything else (n = 7,694 X-rays of 3,307 patients, 22.4% with lung opacity)

| Model | AUROC | inside standing (PA) | inside bedside (AP) | A: sensitivity / specificity | C: sensitivity / specificity |
|---|---|---|---|---|---|
| always_no | 0.500 (0.500 to 0.500) | 0.500 | 0.500 | 0.00 / 1.00 | 0.00 / 1.00 |
| flag_bedside | 0.706 (0.691 to 0.720) | 0.500 | 0.500 | 0.77 / 0.64 | 0.77 / 0.64 |
| metadata_only | 0.706 (0.683 to 0.728) | 0.496 | 0.502 | 0.96 / 0.12 | 0.07 / 0.96 |
| frozen_imagenet | 0.848 (0.834 to 0.863) | 0.819 | 0.776 | 0.97 / 0.41 | 0.39 / 0.94 |
| kids_only | 0.824 (0.810 to 0.838) | 0.795 | 0.727 | 0.95 / 0.44 | 0.29 / 0.94 |
| kids_plus_adults | 0.874 (0.861 to 0.886) | 0.865 | 0.811 | 0.96 / 0.50 | 0.46 / 0.94 |

## Standing X-rays only (PA): n = 4,214, 9.3% with lung opacity

| Model | AUROC | A: sensitivity / specificity | C: sensitivity / specificity |
|---|---|---|---|
| always_no | 0.500 (0.500 to 0.500) | 0.00 / 1.00 | 0.00 / 1.00 |
| flag_bedside | 0.500 (0.500 to 0.500) | 0.00 / 1.00 | 0.00 / 1.00 |
| metadata_only | 0.496 (0.463 to 0.532) | 0.82 / 0.19 | 0.00 / 1.00 |
| frozen_imagenet | 0.819 (0.796 to 0.839) | 0.88 / 0.58 | 0.07 / 1.00 |
| kids_only | 0.795 (0.769 to 0.819) | 0.78 / 0.66 | 0.01 / 1.00 |
| kids_plus_adults | 0.865 (0.846 to 0.882) | 0.85 / 0.70 | 0.16 / 0.99 |
