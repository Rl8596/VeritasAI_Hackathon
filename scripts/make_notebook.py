"""Writes notebooks/pneumoshift_colab.ipynb. Run after changing the pipeline:  python scripts/make_notebook.py"""
import json
from pathlib import Path

REPO = "https://github.com/rl8596/veritasai_hackathon"
CELLS = []


def md(text):
    CELLS.append(dict(cell_type="markdown", metadata={}, source=text.strip("\n")))


def code(text):
    CELLS.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=text.strip("\n")))


md("""
# PneumoShift: does the pneumonia tutorial model work on adults?

The standard pneumonia tutorial dataset contains only children aged 1 to 5 from one hospital.
This notebook trains the tutorial model, then tests it on adults from a different hospital, and measures where it fails.

**Before you start:** `Runtime > Change runtime type > T4 GPU`, then `Runtime > Run all`.

| Mode | What it does | Time on a free Colab GPU |
|---|---|---|
| `QUICK = True` (default) | 6,000 of the 26,684 adult X-rays, 3 training passes | about 10 minutes |
| `QUICK = False` | everything, exactly as in our write-up | about 30 minutes |

Numbers in quick mode differ a little from the write-up because the model sees less data.

**Research code. Not a medical device.**
""")

md("## 1. Setup")
code(f"""
QUICK = True

import os
if not os.path.exists("pneumoshift"):
    !git clone -q {REPO} pneumoshift
%cd pneumoshift
!test -d scripts || git checkout -q claude/loving-hawking-jtiwb3
!pip install -q timm safetensors huggingface_hub pydicom pylibjpeg pylibjpeg-libjpeg scikit-learn pyarrow

import torch
from IPython.display import Image, display
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (training will be slow)")
LIMIT = "--rsna-limit 6000" if QUICK else ""
EPOCHS = 3 if QUICK else 6
""")

md("""
## 2. Data

Two public datasets, both downloaded from the Hugging Face Hub:

- **Children:** Kermany et al. 2018, 5,856 chest X-rays of children aged 1 to 5, Guangzhou, China.
- **Adults:** RSNA Pneumonia Detection Challenge 2018, 26,684 chest X-rays from the NIH Clinical Center, USA, with age, sex and radiologist boxes.

Every image is resized to 256 x 256 pixels. Splits are made **by patient**, so the same person is never in training and test data.
The adult download is 3.9 GB.
""")
code("!python scripts/01_prepare_data.py $LIMIT --workers 2")

md("""
## 3. Check the data before trusting it

Looks for duplicate images across splits and for shortcuts: can something that never looks at the lungs guess the label?
""")
code("""
!python scripts/03_audit.py
!python scripts/04_baselines.py
""")

md("""
## 4. Train the tutorial model (children only)

A ResNet-18 pretrained on ImageNet, loaded from the Hugging Face Hub (`timm/resnet18.tv_in1k`) and fine-tuned.
The decision threshold is set on validation data so that 95% of pneumonia cases are caught. Test data is never used here.
""")
code("!python scripts/02_train.py --name kids_only --reproduce-published --epochs $EPOCHS --workers 2")

md("## 5. Train the comparison model (children + adults)")
code("!python scripts/02_train.py --name kids_plus_adults --reproduce-published --rsna-n 6000 --epochs $EPOCHS --workers 2")

md("""
## 6. The main question: what happens on adults?

Same model, same threshold, held-out test patients. Also: results by age, sex and X-ray view, the
"this patient is unlike my training data" warning, and how many adult X-rays it takes to repair the model.
""")
code("""
!python scripts/05_shift_analysis.py --model kids_only
!python scripts/05_shift_analysis.py --model kids_plus_adults
for name in ["headline", "scores", "subgroups", "ood", "learning_curve"]:
    display(Image(f"results/figures/shift_kids_only_{name}.png", width=760))
display(Image("results/figures/shift_kids_plus_adults_headline.png", width=760))
""")

md("""
## 7. Does the model look at the right place?

Radiologists drew boxes around the pneumonia in the adult X-rays. We check whether the hottest point of the
model's heatmap falls inside a box, and compare with points chosen without any model.
""")
code("""
!python scripts/06_heatmap_check.py --models kids_only kids_plus_adults
for name in ["pointing_test", "adult_examples", "failures_kids_only"]:
    display(Image(f"results/figures/heatmap_{name}.png", width=760))
""")

md("""
## 8. Try one X-ray

This is the same code the demo app runs. To use our published models instead of the ones trained above,
replace the path with `"RL8596/pneumoshift-kids-only"` (downloaded from the Hugging Face Hub).
""")
code("""
import sys; sys.path.insert(0, "src")
from PIL import Image as PILImage
from pneumoshift.inference import Predictor

predictor = Predictor("artifacts/kids_only")
out = predictor.predict(PILImage.open("app/examples/child_pneumonia_1.png"))
print(f"pneumonia score {out['probability']:.0%}  (flag threshold {out['threshold']:.0%})  flagged: {out['flagged']}")
print(f"unlike-training-data warning: {out['ood_level']}")
display(PILImage.fromarray(out["heatmap"]).resize((320, 320)))
""")

md("""
## 9. Start the demo app

Prints a public link (valid for about a week while this cell runs). The app loads our published models from the Hugging Face Hub.
""")
code("""
!pip install -q gradio==6.32.0
%env PNEUMOSHIFT_SHARE=1
!python app/app.py
""")

md("""
## Sources

- Kermany D, Zhang K, Goldbaum M. Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification. Mendeley Data V2, 2018. CC BY 4.0. Hugging Face copy: `hf-vision/chest-xray-pneumonia`.
- Radiological Society of North America. RSNA Pneumonia Detection Challenge, 2018. Images from the NIH Clinical Center (Wang et al. 2017, ChestX-ray8). Hugging Face copy: `kellly/RSNA-Pneumonia-Detection`.
- Shih G et al. Augmenting the NIH Chest Radiograph Dataset with Expert Annotations of Possible Pneumonia. Radiology: Artificial Intelligence 2019.
- He K et al. Deep Residual Learning for Image Recognition, 2016. Weights: `timm/resnet18.tv_in1k` (torchvision, ImageNet-1k).
- Selvaraju RR et al. Grad-CAM, 2017. Lee K et al. A Simple Unified Framework for Detecting Out-of-Distribution Samples, 2018. Guo C et al. On Calibration of Modern Neural Networks, 2017.

Full list in the repository README.
""")

nb = dict(cells=CELLS, nbformat=4, nbformat_minor=5,
          metadata=dict(accelerator="GPU", colab=dict(provenance=[], gpuType="T4"),
                        kernelspec=dict(name="python3", display_name="Python 3"), language_info=dict(name="python")))
for c in CELLS:
    c["source"] = c["source"].splitlines(keepends=True)
out = Path(__file__).resolve().parents[1] / "notebooks" / "pneumoshift_colab.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, f"({len(CELLS)} cells)")
