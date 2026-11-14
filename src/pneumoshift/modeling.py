"""Build, save, load and run the classifier.

A saved model is a folder (or a Hugging Face Hub repo) with:
    model.safetensors   weights
    config.json         architecture name, thresholds, calibration, metrics, training info
    ood.npz             (optional) statistics for the "unlike my training data" warning
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import config as C
from .preprocess import to_model_input


def build_model(arch: str = C.DEFAULT_ARCH, pretrained: bool = True) -> nn.Module:
    """A timm image model with one output (the pneumonia logit).

    With pretrained=True the ImageNet weights are downloaded from the Hugging Face Hub.
    """
    import timm

    return timm.create_model(arch, pretrained=pretrained, num_classes=1, in_chans=3)


def save_checkpoint(model: nn.Module, out_dir, config: dict) -> Path:
    """config must contain at least {"arch": ...}. Everything else is free-form and
    is shown by the demo app (thresholds, temperature, metrics, training data)."""
    from safetensors.torch import save_file

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(out_dir / "model.safetensors"))
    cfg = dict(arch=C.DEFAULT_ARCH, img_size=C.IMG_SIZE, store_size=C.STORE_SIZE, class_names=list(C.CLASS_NAMES),
               temperature=1.0, threshold=0.5)
    cfg.update(config)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    return out_dir


def resolve_dir(path_or_repo) -> Path:
    """A local folder, or a Hugging Face Hub model repo id such as 'user/name'."""
    p = Path(path_or_repo)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(path_or_repo)))


def load_checkpoint(path_or_repo, device: str = "cpu"):
    """-> (model in eval mode, config dict)."""
    from safetensors.torch import load_file

    d = resolve_dir(path_or_repo)
    cfg = json.loads((d / "config.json").read_text())
    model = build_model(cfg["arch"], pretrained=False)
    model.load_state_dict(load_file(str(d / "model.safetensors")))
    return model.to(device).eval(), cfg


def forward_with_embedding(model: nn.Module, x: torch.Tensor):
    """-> (logits [B], embedding [B, D]). The embedding is the pooled feature vector
    right before the final linear layer."""
    feats = model.forward_features(x)
    emb = model.forward_head(feats, pre_logits=True)
    logits = model.get_classifier()(emb).squeeze(1)
    return logits, emb


@torch.no_grad()
def predict(model: nn.Module, X, indices=None, batch_size: int = 128, device: str = "cpu",
            return_embeddings: bool = False):
    """Run the model over stored uint8 images X[indices] (centre crop, no augmentation).

    -> logits [N] (numpy), and embeddings [N, D] if return_embeddings.
    Probability of pneumonia = sigmoid(logit / temperature).
    """
    model.eval()
    indices = np.arange(len(X)) if indices is None else np.asarray(indices)
    logits, embs = [], []
    for i in range(0, len(indices), batch_size):
        idx = np.sort(indices[i:i + batch_size])            # sorted reads are faster on a memmap
        order = np.argsort(np.argsort(indices[i:i + batch_size]))
        x = to_model_input(np.asarray(X[idx])).to(device)
        lg, em = forward_with_embedding(model, x)
        logits.append(lg.float().cpu().numpy()[order])
        if return_embeddings:
            embs.append(em.float().cpu().numpy()[order])
    logits = np.concatenate(logits)
    return (logits, np.concatenate(embs)) if return_embeddings else logits


def sigmoid(z, temperature: float = 1.0):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64) / temperature))
