# PneumoShift

**A second look at a chest X-ray for pneumonia, for a clinic without a radiologist, that says when it is out of its depth.**

Try it: **https://huggingface.co/spaces/RL8596/pneumoshift** (runs in your browser, nothing is uploaded).
Research demo. Not a medical device. Not for diagnosis.

Track: AI + Health. Models: [children-only](https://huggingface.co/RL8596/pneumoshift-kids-only), [children + adults](https://huggingface.co/RL8596/pneumoshift-kids-plus-adults).

---

## One-page summary

### The problem and the user

A small clinic takes a chest X-ray, but no radiologist is there to read it. In Malawi there are 3 radiologists for
19.7 million people [15]. A tool that sorts X-rays into "look at this one first" could save hours for the patients
who need them most.

The standard way to build such a tool is a well-known tutorial: train a network on a public dataset of chest
X-rays and report over 90% accuracy. That dataset holds **only children aged 1 to 5 from one hospital in China** [1].
Yet in 2021 more than twice as many people aged 70 and over died of lung infections as children under 5 [14].

### Our question

**What happens when the tutorial model meets adults from another hospital, and can the tool notice it?**

Three papers in 2026 already showed that such a model gets worse on adults [9, 10, 11]. We repeat that result and
do not claim it. What we add, to our knowledge, is the part a clinic needs: *how* it fails, a built-in warning,
and what it takes to repair it.

### What we built

- **Two models** (ResNet-18, a standard image network): one trained on the 4,578 children's X-rays only, one on the
  children plus 6,000 adult X-rays from a US hospital [3].
- **A demo** that shows, for both models side by side: a flag, a score, a heatmap of where the model looked, and
  two warnings (the X-ray looks unlike the training X-rays; the patient's age group was never in the training data).
- **Tests on patients the models never saw**: 624 children and 7,694 adults.

### Results

AUROC measures how well a model ranks sick above healthy: 0.5 is guessing, 1.0 is perfect.
"Adults" means: lung opacity against everyone else, including adults who are sick with something else.

| | Children, same hospital | Adults, other hospital |
|---|---|---|
| Children-only model (the tutorial) | **0.975** | 0.824 |
| Children + 6,000 adults | 0.962 | **0.874** |
| *Baseline:* raw pixels + logistic regression, trained on children | 0.847 | 0.727 |
| *Baseline:* untrained-for-X-rays ImageNet network + logistic regression, trained on children | 0.953 | 0.752 |
| *Baseline:* the same ImageNet network + logistic regression, trained on adults | | 0.848 |
| *Baseline:* looks at no picture, only at "bedside or standing X-ray" | | 0.706 |

### What we learned

1. **The headline number hides the failure.** The children-only model still ranks adults fairly well (0.824). But
   used as built, with its threshold, it is right for 74.7% of adults, while "nobody has pneumonia" is right for
   77.6% and catches no one. Accuracy is the wrong yardstick when 78% of patients are negative, so we report
   sensitivity (cases caught) and specificity (non-cases cleared) everywhere.
2. **The failure has a shape.** 94% of the false alarms are adults who are sick with *something else*. And one
   threshold does not fit both kinds of X-ray: the children-only model catches 93% of cases on bedside X-rays but
   only 28% on standing X-rays, which are the ones a small clinic takes.
3. **The warning notices the wrong hospital, not the wrong answer.** It fires for 7% of the children's test X-rays
   and 81% of the adults'. But it cannot tell which adult answers are wrong.
4. **Adult data repairs part of it, and a simple baseline gets most of the way.** 6,000 adult X-rays lift the adult
   score from 0.824 to 0.874. A network that was never fine-tuned reaches 0.848 with the same adults. So the honest
   gain of our fine-tuning over that baseline is 0.025 (95% interval 0.019 to 0.032).
5. **The heatmaps do not show where the disease is.** We tested them against radiologists' boxes: the hottest point
   lies inside a box for 25% (children-only) and 52% (children + adults) of X-rays. Always pointing at the most
   common spot, without looking at the picture, scores 57%.
6. **We found a leak in the usual way of splitting the adult data, in our own first attempt.** The dataset's
   "patientId" is an exam number. A random split put the same people in training and test (5,493 of 8,006 test
   exams). We split by the real patient number instead.

### What did not work

- Retraining only the last layer with adult X-rays barely helps (0.821 to 0.839 with all 15,142).
- The repaired model pays for adults with children: at its threshold it clears only 46% of healthy children
  (the children-only model: 87%).
- Neither model beats "always no" in accuracy at the 95%-sensitivity screening threshold.
- A Gradio demo on Hugging Face was refused, so we rebuilt the demo to run in the browser.

### Limits

Two hospitals, public data, one training run per published model. "Lung opacity" on the adult X-rays is a
radiologist's finding on the picture, not a confirmed pneumonia [5]. No authority has checked this tool.
A second generation of models with more adult data, three random seeds and a comparison plan fixed in advance
([docs/V2_PROTOCOL.md](docs/V2_PROTOCOL.md)) is described in section 6.

---

## Details

### 1. Data

| | Children | Adults |
|---|---|---|
| Source | Kermany et al. 2018, Guangzhou Women and Children's Medical Center [1, 4]. Licence CC BY 4.0. Hub: `hf-vision/chest-xray-pneumonia` | RSNA Pneumonia Detection Challenge 2018, X-rays of the NIH Clinical Center, USA [3, 5, 6]. Hub: `kellly/RSNA-Pneumonia-Detection` |
| Size | 5,856 X-rays, ages 1 to 5 | 26,684 exams of 11,452 patients, with age, sex, view and radiologists' boxes |
| Labels | normal, pneumonia | normal; no lung opacity but not normal ("sick with something else"); lung opacity |
| Our split, by patient | train 4,578 / validation 654 / official test 624 | training pool 16,010 / validation 2,667 / test 8,007 (7,694 adults) |

- **Validation data** is used for every choice (which training round to keep, the threshold). **Test data** is
  only scored. Nothing was tuned on test data.
- The adult dataset contains 1,281 X-rays of minors. They are filtered out by the age in the file header and used
  for one check only (section 4).
- We examined a third dataset ("Epic Chittagong" on the Hub) as independent test data and rejected it: it contains
  copies of the children's images.

### 2. Models and the threshold rule

- ResNet-18 pretrained on ImageNet (`timm/resnet18.tv_in1k` from the Hugging Face Hub), fine-tuned whole, one output.
  Images are made grey, cropped to a square and reduced to 224 pixels. Light augmentation, no left-right flips
  (the heart is on one side).
- **Threshold rule, fixed before testing:** the highest threshold that still catches 95% of the pneumonia cases
  in the validation X-rays. A screening tool should miss few cases, and pays for that with false alarms.
- The score is not the chance that the patient has pneumonia. On adults the children-only model's scores are far
  off (calibration error 0.40; 0 would be perfect), the second model's are close (0.03).

### 3. How the children-only model fails on adults

Test patients only. Threshold 0.985, chosen on children's validation X-rays.

| Test group | X-rays | AUROC | Cases caught | Non-cases cleared | Right answers | "Always the same answer" |
|---|---|---|---|---|---|---|
| Children 1 to 5, same hospital | 624 | 0.975 | 97.9% | 86.7% | 93.7% | 62.5% |
| Adults, lung opacity vs normal | 4,242 | 0.953 | 78.1% | 96.0% | 88.7% | 59.3% |
| Adults, lung opacity vs everyone else | 7,694 | 0.824 | 78.1% | 73.7% | 74.7% | 77.6% |

- **By diagnosis:** it flags 4% of normal adults, 42% of adults who are sick with something else, 78% of adults
  with lung opacity.
- **By X-ray view:** standing (PA): 28% of cases caught, 95% of non-cases cleared. Bedside (AP): 93% caught, 36%
  cleared. Sicker patients are X-rayed in bed, so the view alone predicts the label (0.706).
- **By age and sex:** small differences (AUROC 0.841 at 18 to 39, 0.802 at 60 and over; women 0.818, men 0.829).

The second model (children + 6,000 adults, threshold 0.157 chosen on children's and adult validation X-rays together):

| Test group | AUROC | Cases caught | Non-cases cleared | Right answers |
|---|---|---|---|---|
| Children 1 to 5 | 0.962 | 100% | 45.7% | 79.6% |
| Adults, lung opacity vs normal | 0.979 | 91.1% | 94.8% | 93.3% |
| Adults, lung opacity vs everyone else | 0.874 | 91.1% | 65.0% | 70.8% |

It still flags 57% of adults who are sick with something else. Telling lung opacity from other lung disease is the
hard part, and more adult data did not solve it.

**A fairer comparison of the two models.** Each model above sits at a different point: the first catches 78% of
adult cases, the second 91%. With the threshold chosen on adult validation X-rays, so that both catch about 95%:

| Model | Adults: cases caught | Non-cases cleared |
|---|---|---|
| Children-only | 95% | 44% |
| Children + 6,000 adults | 96% | 50% |
| Frozen ImageNet network + logistic regression, trained on adults | 97% | 41% |

### 4. The "unlike my training data" warning

For every X-ray the model produces 512 internal numbers. We measure how far they lie from those of the training
X-rays (Mahalanobis distance [16]) and warn above the level that 95% of the validation X-rays stay below.

- Children-only model: fires for 7.2% of children's test X-rays and 80.6% of adults'.
- **It detects the hospital, not the age.** It also fires for 97% of the other hospital's patients under 18.
- **It is not an error detector.** Among adults it does not single out the wrong answers.
- For the second model the picture turns around: the warning fires for 24% of children and 1% of adults. It always
  means "unlike this model's own training mix".

### 5. Heatmaps against radiologists' boxes

On 1,500 adult test X-rays with boxes drawn by radiologists, we asked: does the hottest point of the heatmap
(Grad-CAM [17]) lie inside a box?

| Random point | Image centre | Always the most common spot | Children-only model | Children + adults model |
|---|---|---|---|---|
| 15% | 10% | 57% | 25% | 52% |

The heatmap has a 7 by 7 grid, so it is coarse. The demo says that it shows what influenced the model and is not a
map of the disease.

### 6. Second generation of models (in progress)

The team asked for a stronger repaired model. Before training anything we wrote down what would be trained, which
model would be the headline, how it would be compared, and what "good" means:
[docs/V2_PROTOCOL.md](docs/V2_PROTOCOL.md). In short:

- One recipe, 22 runs on a GPU: children + 0 / 300 / 1,000 / 3,000 / 6,000 / all 15,142 adult X-rays, three
  random seeds for the main points, and an adults-only model.
- The headline model is fixed by name. No model is picked because of its test result.
- Test results are computed once, by `scripts/10_v2_report.py`, with intervals that resample patients.
- We wrote in advance that we do not expect it to beat "always no" at the screening threshold.

*Results: to be added when the runs are complete.*

### 7. Checks on the data

- No image is shared between training and test, or between the two datasets. 32 exact duplicates sit inside
  single splits of the children's data.
- **Shortcuts in the children's data:** the image size alone predicts the label (AUROC 0.879), the brightness
  alone 0.871. A model can look good there for the wrong reasons.
- A simple classifier tells the two hospitals apart with AUROC 0.998, and still 0.951 for children of the same age.
- On the children's data, validation looks better than test (even raw pixels reach 0.99 on validation and 0.847
  on test), so the official test set differs from the training pool.

### 8. The demo

`web/` is a static page. Both models run on the visitor's device with ONNX Runtime Web, so the picture never
leaves it. The page repeats the Python preprocessing step for step. `tests/web/parity.mjs` opens the page in a
headless browser and compares it with the Python code on 15 pictures: the model input is identical byte for byte,
and every decision and warning matches. `app/` holds the same demo as a Gradio app.

### 9. How to run

```bash
pip install -r requirements.txt
python scripts/01_prepare_data.py     # downloads about 5 GB from the Hugging Face Hub
python scripts/hub_models.py pull     # the published models and their per-image scores
```

| Step | Command |
|---|---|
| Train | `python scripts/02_train.py --name my_model` (add `--rsna-n 6000` for adult X-rays) |
| Data checks, baselines | `python scripts/03_audit.py`, `python scripts/04_baselines.py` |
| Children vs adults | `python scripts/05_shift_analysis.py --model <name>` |
| Heatmap test | `python scripts/06_heatmap_check.py --models kids_only kids_plus_adults` |
| Final comparison | `python scripts/10_v2_report.py test --v1-only` |
| Demo | open `web/index.html` through a local web server, or `python app/app.py` |
| Notebook | `notebooks/pneumoshift_colab.ipynb` runs the pipeline in Colab |

Training one model takes about an hour on a CPU and a few minutes on a GPU.

### 10. What is ours and what is not

- **Not ours:** the two datasets, the pretrained network, the finding that a children-trained model is worse on
  adults [9, 10, 11], and the general lesson that medical models can fail in a new hospital [7, 8].
- **Ours, to our knowledge:** the breakdown of the failure by X-ray view and by diagnosis; the comparison against
  baselines that never look at the picture; the warning tested for what it can and cannot do; the heatmap test
  against radiologists' boxes with no-skill reference points; the patient-level split of the adult data; and a
  demo that runs both models in the browser.

### 11. Use of AI assistance

<!-- TEAM: check the hackathon rules on AI assistance and adapt this paragraph before submitting. -->
We used an AI coding assistant (Claude, by Anthropic) throughout: to write and review most of the code, to run
the experiments, and to draft this text. We chose the question, the user and the direction, checked the results,
and are responsible for every statement here.

### 12. Sources

Datasets
1. Kermany D, Zhang K, Goldbaum M. Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification. Mendeley Data, V2, 2018. doi:10.17632/rscbjbr9sj.2. Licence CC BY 4.0.
2. Mooney P. Chest X-Ray Images (Pneumonia). Kaggle, 2018 (copy of [1]).
3. Radiological Society of North America. RSNA Pneumonia Detection Challenge (2018). Images and annotations.

Papers on the datasets
4. Kermany DS, Goldbaum M, Cai W, et al. Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. Cell. 2018;172(5):1122-1131.e9. doi:10.1016/j.cell.2018.02.010
5. Shih G, Wu CC, Halabi SS, et al. Augmenting the National Institutes of Health Chest Radiograph Dataset with Expert Annotations of Possible Pneumonia. Radiology: Artificial Intelligence. 2019;1(1):e180041. doi:10.1148/ryai.2019180041
6. Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks. IEEE CVPR. 2017:3462-3471. doi:10.1109/CVPR.2017.369

Generalisation and shortcuts
7. Zech JR, Badgeley MA, Liu M, et al. Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs. PLOS Medicine. 2018;15(11):e1002683. doi:10.1371/journal.pmed.1002683
8. Roberts M, Driggs D, Thorpe M, et al. Common pitfalls and recommendations for using machine learning to detect and prognosticate for COVID-19 using chest radiographs and CT scans. Nature Machine Intelligence. 2021;3(3):199-217. doi:10.1038/s42256-021-00307-0
9. Javier-Tarapaqui J, Vega-Huerta H, Guerra-Grados L, et al. Deep Learning Architecture Comparison for Pneumonia Detection in Pediatric Chest X-Rays: Ensemble Learning and Cross-Population External Evaluation. Applied Sciences. 2026;16(21):11063. doi:10.3390/app162111063
10. Gómez-Vela F, López-Fernandez A, Divina F, García-Torres M. Performance-interpretability trade-offs and generalization in deep learning for pneumonia detection: A benchmarking study. Applied Intelligence. 2026;56(14):414. doi:10.1007/s10489-026-07398-5
11. Neo RYS. Benchmarking knowledge distillation for lightweight pneumonia detection: a multi-seed calibration study on PneumoniaMNIST. BMC Research Notes. 2026;19(1):270. doi:10.1186/s13104-026-07793-8
12. Geirhos R, Jacobsen J-H, Michaelis C, et al. Shortcut learning in deep neural networks. Nature Machine Intelligence. 2020;2(11):665-673. doi:10.1038/s42256-020-00257-z

Impact
13. World Health Organization. Pneumonia in children (fact sheet). 11 November 2022.
14. GBD 2021 Lower Respiratory Infections and Antimicrobial Resistance Collaborators. Global, regional, and national incidence and mortality burden of non-COVID-19 lower respiratory infections and aetiologies, 1990-2021. Lancet Infectious Diseases. 2024;24(9):974-1002. doi:10.1016/S1473-3099(24)00176-2
15. Kawooya MG, Kisembo HN, Remedios D, et al. An Africa point of view on quality and safety in imaging. Insights into Imaging. 2022;13:58. doi:10.1186/s13244-022-01203-w

Methods and software
16. Lee K, Lee K, Lee H, Shin J. A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks. NeurIPS 2018.
17. Selvaraju RR, Cogswell M, Das A, et al. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. IEEE ICCV. 2017.
18. He K, Zhang X, Ren S, Sun J. Deep Residual Learning for Image Recognition. IEEE CVPR. 2016. Weights: `timm/resnet18.tv_in1k` (Wightman R. PyTorch Image Models).
19. Guo C, Pleiss G, Sun Y, Weinberger KQ. On Calibration of Modern Neural Networks. ICML 2017.
20. ONNX Runtime Web (Microsoft, MIT licence); PyTorch; scikit-learn; Hugging Face Hub.
