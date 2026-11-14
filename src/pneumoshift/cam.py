"""Grad-CAM: where in the X-ray did the model look?

Works for any timm CNN (uses forward_features / forward_head), no extra library.
"""
import numpy as np
import torch
import torch.nn.functional as F


def grad_cam(model, x: torch.Tensor):
    """x: model input [B, 3, H, W]. -> (cam [B, H, W] numpy in 0..1, logits [B] numpy).

    The map shows evidence FOR pneumonia. For each image it is divided by its own
    maximum, so it shows where the model looked, not how confident it was.
    """
    model.eval()
    with torch.enable_grad():
        feats = model.forward_features(x)                    # [B, C, h, w]
        logits = model.forward_head(feats).squeeze(1)        # [B]
        grads = torch.autograd.grad(logits.sum(), feats)[0]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * feats).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
    return cam.detach().cpu().numpy(), logits.detach().cpu().numpy()


def overlay(gray: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """gray uint8 [H, W] + cam [H, W] in 0..1 -> RGB uint8 [H, W, 3] heatmap overlay.

    Weak activations stay transparent so the X-ray remains readable.
    """
    from matplotlib import colormaps

    heat = colormaps["inferno"](cam)[..., :3]
    base = np.repeat(gray[..., None].astype(np.float32) / 255.0, 3, axis=2)
    a = (alpha * np.clip(cam, 0, 1) ** 0.7)[..., None]
    return (np.clip(base * (1 - a) + heat * a, 0, 1) * 255).astype(np.uint8)
