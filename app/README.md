---
title: PneumoShift
emoji: 🩻
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.32.0
python_version: "3.13"
app_file: app.py
pinned: false
license: mit
short_description: Pneumonia X-ray screening that warns when it is out of its depth
---

# PneumoShift

A second look at a chest X-ray for pneumonia, for a clinic without a radiologist.

Most pneumonia tutorials train on one dataset of children aged 1 to 5 from a single hospital.
This demo compares such a children-only model with a model that also learned from adults, and
it tells you when a result should not be trusted:

- it shows where each model looked (Grad-CAM heatmap),
- it warns when an X-ray looks unlike the X-rays the model learned from,
- it warns when the patient's age group was never part of the model's training.

**Research demo. Not a medical device. Not for diagnosis.**

## Settings (Space variables, all optional)

| Variable | Default | Meaning |
|---|---|---|
| `PNEUMOSHIFT_KIDS_ONLY_REPO` | `RL8596/pneumoshift-kids-only` | Hub model repo of the children-only model |
| `PNEUMOSHIFT_KIDS_PLUS_ADULTS_REPO` | `RL8596/pneumoshift-kids-plus-adults` | Hub model repo of the children + adults model |
| `PNEUMOSHIFT_THREADS` | not set | Limit PyTorch to this many CPU threads |

The model repos must be public (or the Space needs an `HF_TOKEN` secret). A model that cannot be
loaded is skipped; the app runs with whichever models are available.

## Data and credits

- Example X-rays of children: Kermany, Zhang and Goldbaum (2018), "Labeled Optical Coherence Tomography (OCT)
  and Chest X-Ray Images for Classification", Mendeley Data, v2. Licence CC BY 4.0.
- Adult X-rays used for training and testing: RSNA Pneumonia Detection Challenge (2018).
- Backbone: ResNet-18 pretrained on ImageNet (`timm/resnet18.tv_in1k`).
