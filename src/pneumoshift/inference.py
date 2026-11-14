"""From an uploaded picture to a result: the one code path the demo app uses.

    p = Predictor("artifacts/kids_only")     # a local folder or a Hugging Face Hub repo id
    r = p.predict(pil_image)                 # see Predictor.predict for the keys of r

One forward pass gives the probability, the embedding (for the "unlike my training
data" check) and the Grad-CAM heatmap. The picture goes through
preprocess.pil_to_model_input, exactly like the test images.
"""
from __future__ import annotations

import threading
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from .cam import grad_cam, overlay
from .modeling import forward_with_embedding, load_checkpoint, resolve_dir, sigmoid
from .ood import MahalanobisOOD
from .preprocess import pil_to_model_input

OOD_LEVELS = ("ok", "unusual", "very unusual")

_ALPHA_MODES = ("RGBA", "LA", "PA", "RGBa", "La")
_PLAIN_MODES = ("L", "RGB")
_DEEP_MODES = ("I;16", "I;16B", "I;16L", "I", "F")     # the modes preprocess.to_square_gray rescales itself


def to_pil(image) -> Image.Image:
    """Whatever a user hands us -> a PIL image that the shared preprocessing treats correctly.

    Accepts a PIL image, a file path or a numpy array. For an ordinary grayscale or
    RGB picture this changes nothing. It only tidies up the odd cases:
      - photos taken with a rotated phone are turned upright (EXIF orientation),
      - transparent pixels become black (the colour of "nothing" on an X-ray),
      - 16-bit and float images are kept as they are, preprocessing rescales them,
      - palette, CMYK and other unusual modes become RGB.
    Raises ValueError with a readable message if the picture cannot be read.
    """
    if isinstance(image, (str, Path)):
        try:
            image = Image.open(image)
        except Image.DecompressionBombError as e:
            raise ValueError("The image is too large to open safely.") from e
        except Exception as e:
            raise ValueError("The file could not be read as an image.") from e
    elif isinstance(image, np.ndarray):
        image = _array_to_pil(image)
    elif not isinstance(image, Image.Image):
        raise ValueError(f"Expected an image, got {type(image).__name__}.")

    try:
        image.load()                                   # decode now, so a broken file fails here
        if getattr(image, "n_frames", 1) > 1:          # animated GIF, multi-page TIFF: first frame
            image.seek(0)
            image = image.copy()
        image = ImageOps.exif_transpose(image) or image
    except Image.DecompressionBombError as e:
        raise ValueError("The image is too large to open safely.") from e
    except Exception as e:
        raise ValueError("The image file is damaged or incomplete.") from e

    if image.width < 1 or image.height < 1:
        raise ValueError("The image is empty.")

    try:
        if image.mode in _ALPHA_MODES or (image.mode == "P" and "transparency" in image.info):
            rgba = image.convert("RGBA")
            black = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
            image = Image.alpha_composite(black, rgba).convert("RGB")
        elif image.mode == "F":
            a = np.nan_to_num(np.asarray(image, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            image = Image.fromarray(a)
        elif image.mode.startswith("I;16") and image.mode not in _DEEP_MODES:
            image = image.convert("I")                 # e.g. "I;16N"
        elif image.mode not in _PLAIN_MODES + _DEEP_MODES:
            image = image.convert("RGB")               # "1", "P", "CMYK", "YCbCr", "LAB", "HSV", ...
    except Exception as e:
        raise ValueError(f"Images of type '{image.mode}' are not supported.") from e
    return image


def _array_to_pil(a: np.ndarray) -> Image.Image:
    a = np.asarray(a)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[..., 0]
    if a.ndim not in (2, 3) or a.size == 0 or (a.ndim == 3 and a.shape[2] not in (3, 4)):
        raise ValueError(f"Cannot read an array of shape {a.shape} as an image.")
    if a.dtype == np.uint8:
        return Image.fromarray(a)
    a = np.nan_to_num(a.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if a.ndim == 2:
        return Image.fromarray(a)                      # mode "F", rescaled by preprocessing
    lo, hi = float(a.min()), float(a.max())
    return Image.fromarray(((a - lo) / (hi - lo + 1e-8) * 255.0).astype(np.uint8))


class Predictor:
    """A trained model plus everything needed to explain one prediction.

    Attributes: config (the model's config dict), temperature, threshold,
    ood (a MahalanobisOOD, or None if the model folder has no ood.npz).
    """

    def __init__(self, path_or_repo, device: str = "cpu"):
        self.source = str(path_or_repo)
        self.dir = resolve_dir(path_or_repo)
        self.device = device
        self.model, self.config = load_checkpoint(self.dir, device)
        self.temperature = float(self.config.get("temperature", 1.0))
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError(f"temperature must be a positive number, got {self.temperature!r}")
        self.threshold = float(self.config.get("threshold", 0.5))

        self.ood = None
        ood_path = self.dir / "ood.npz"
        if ood_path.exists():
            try:
                self.ood = MahalanobisOOD.load(ood_path)
                dim = getattr(self.model, "num_features", self.ood.means.shape[1])
                if self.ood.means.shape[1] != dim or len(self.ood.ref_scores) == 0:
                    raise ValueError(f"ood.npz does not fit this model (expected embedding size {dim})")
            except Exception as e:      # a broken ood.npz must not take the whole model down
                warnings.warn(f"{ood_path}: ignored, the out-of-distribution check is off ({e})")
                self.ood = None
        self._lock = threading.Lock()   # one prediction at a time per model (hook + CPU threads)

    # ------------------------------------------------------------------
    def _forward(self, x: torch.Tensor):
        """One forward pass -> (cam [H, W], logit, embedding [D]).

        cam.grad_cam already runs the network once to get the heatmap and the logit.
        The embedding is the input of the final linear layer in that same pass, so
        we catch it with a hook instead of running the network a second time.
        """
        caught = []
        hook = self.model.get_classifier().register_forward_pre_hook(
            lambda module, inputs: caught.append(inputs[0].detach()))
        try:
            cams, logits = grad_cam(self.model, x)
        finally:
            hook.remove()
        if len(caught) == 1:
            emb = caught[0]
        else:                           # unusual architecture: fall back to a second pass
            with torch.no_grad():
                _, emb = forward_with_embedding(self.model, x)
        return cams[0], float(logits[0]), emb[0].float().cpu().numpy()

    def predict(self, image) -> dict:
        """image: PIL image (a file path or numpy array also works). -> dict with

        probability     float, temperature-calibrated probability of pneumonia
        threshold       float, the model's decision threshold
        flagged         bool, probability >= threshold
        heatmap         RGB uint8 [224, 224, 3], Grad-CAM overlay (where the model looked)
        input_image     uint8 [224, 224], the picture exactly as the model saw it
        ood_score       float, distance from the training data (None without ood.npz)
        ood_percentile  float 0..100, share of held-out training-like X-rays that looked
                        MORE familiar than this one (None without ood.npz)
        ood_level       "ok" | "unusual" (above the 95th percentile) | "very unusual"
                        (above the 99th percentile), None without ood.npz
        config          the model's config dict
        logit, cam      the raw logit and the raw Grad-CAM map [224, 224] in 0..1
        """
        x, gray = pil_to_model_input(to_pil(image))
        gray = np.ascontiguousarray(gray)
        with self._lock:
            cam, logit, emb = self._forward(x.to(self.device))
        if not np.isfinite(logit):
            raise ValueError("The model returned an invalid number for this image.")

        probability = float(sigmoid(logit, self.temperature))
        cam = np.nan_to_num(cam, nan=0.0)
        out = dict(
            probability=probability,
            threshold=self.threshold,
            flagged=bool(probability >= self.threshold),
            heatmap=overlay(gray, cam),
            input_image=gray,
            ood_score=None, ood_percentile=None, ood_level=None,
            config=self.config,
            logit=logit,
            cam=cam,
        )
        if self.ood is not None:
            score = float(self.ood.score(emb)[0])
            out["ood_score"] = score
            out["ood_percentile"] = float(np.ravel(self.ood.percentile(score))[0])
            t = self.ood.thresholds
            out["ood_level"] = "very unusual" if score > t["p99"] else "unusual" if score > t["p95"] else "ok"
        return out
