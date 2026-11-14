"""The single definition of how an X-ray becomes model input.

Training, evaluation and the demo app all go through these two functions, so a
picture uploaded in the demo is treated exactly like the test images.
"""
import numpy as np
import torch
from PIL import Image

from .config import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, STORE_SIZE


def to_square_gray(img: Image.Image, size: int = STORE_SIZE) -> np.ndarray:
    """PIL image (any mode or size) -> uint8 array [size, size].

    Grayscale, centre-crop to a square (no stretching, so anatomy keeps its
    proportions), then downscale with antialiasing.
    """
    if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):  # 16-bit / float X-rays
        a = np.asarray(img).astype(np.float32)
        lo, hi = float(a.min()), float(a.max())
        a = (a - lo) / (hi - lo + 1e-8) * 255.0
        img = Image.fromarray(a.astype(np.uint8))
    img = img.convert("L")
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    img = img.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def center_crop(arr: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Centre-crop the last two dims of a [..., H, W] array to [..., size, size]."""
    h, w = arr.shape[-2:]
    top, left = (h - size) // 2, (w - size) // 2
    return arr[..., top:top + size, left:left + size]


def normalize(t: torch.Tensor) -> torch.Tensor:
    """uint8 or float [0,255] tensor [B, 1, H, W] -> normalised float [B, 3, H, W]."""
    t = t.float() / 255.0
    t = t.expand(-1, 3, -1, -1)
    mean = torch.tensor(IMAGENET_MEAN, device=t.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=t.device).view(1, 3, 1, 1)
    return (t - mean) / std


def to_model_input(arr: np.ndarray) -> torch.Tensor:
    """uint8 array [H, W] or [B, H, W] (stored 256 px images) -> float tensor [B, 3, 224, 224]."""
    if arr.ndim == 2:
        arr = arr[None]
    if arr.shape[-1] != IMG_SIZE:
        arr = center_crop(arr, IMG_SIZE)
    t = torch.from_numpy(np.ascontiguousarray(arr))[:, None]
    return normalize(t)


def pil_to_model_input(img: Image.Image):
    """PIL image -> (model input tensor [1, 3, 224, 224], the 224 px uint8 image the model sees)."""
    gray = center_crop(to_square_gray(img), IMG_SIZE)
    return to_model_input(gray), gray
