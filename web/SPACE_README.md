---
title: PneumoShift
emoji: 🩻
colorFrom: blue
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Chest X-ray pneumonia check that knows its limits
models:
  - RL8596/pneumoshift-kids-only
  - RL8596/pneumoshift-kids-plus-adults
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

## How it runs

This is a static page. Both models run in your browser with ONNX Runtime Web, so the picture you
choose never leaves your device. The model files (about 45 MB each) are downloaded once from the
model repositories on the Hugging Face Hub and then kept by your browser.

## Data and credits

- Example X-rays of children: Kermany, Zhang and Goldbaum (2018), "Labeled Optical Coherence Tomography (OCT)
  and Chest X-Ray Images for Classification", Mendeley Data, v2. Licence CC BY 4.0.
- Adult X-rays used for training and testing: RSNA Pneumonia Detection Challenge (2018).
- Backbone: ResNet-18 pretrained on ImageNet (`timm/resnet18.tv_in1k`).
- Runtime: ONNX Runtime Web (MIT licence), loaded from jsDelivr.
- Typeface: IBM Plex Sans and IBM Plex Mono (SIL Open Font License 1.1, `fonts/OFL.txt`).
