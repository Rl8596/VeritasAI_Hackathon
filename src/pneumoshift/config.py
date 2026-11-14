"""Paths and constants shared by every script, the notebook and the demo app."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Raw downloads and processed arrays (not in git). Override with PNEUMOSHIFT_CACHE.
CACHE_DIR = Path(os.environ.get("PNEUMOSHIFT_CACHE", REPO_ROOT / "data_cache"))
PROCESSED_DIR = CACHE_DIR / "processed"
# Trained models (not in git, pushed to the Hugging Face Hub instead).
ARTIFACTS_DIR = Path(os.environ.get("PNEUMOSHIFT_ARTIFACTS", REPO_ROOT / "artifacts"))
# Small result files (json, csv, png) that ARE committed to git.
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Hugging Face sources
KERMANY_REPO = "hf-vision/chest-xray-pneumonia"   # children 1-5, Guangzhou (Kermany et al. 2018), CC BY 4.0
RSNA_REPO = "kellly/RSNA-Pneumonia-Detection"     # mostly adults, USA (RSNA 2018 challenge, NIH images)
RSNA_ZIP = "rsna-pneumonia-detection-challenge.zip"
RSNA_NIH_MAPPING_URL = "https://s3.amazonaws.com/east1.public.rsna.org/AI/2018/pneumonia-challenge-dataset-mappings_2018.json"

# Images are stored as 256x256 8-bit grayscale and fed to the model as 224x224.
STORE_SIZE = 256
IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Pretrained backbone from the Hugging Face Hub (timm/resnet18.tv_in1k, BSD-3 licence).
DEFAULT_ARCH = "resnet18.tv_in1k"

SEED = 2026
CLASS_NAMES = ("normal", "pneumonia")
ADULT_AGE = 18
