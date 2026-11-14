"""Fine-tune the pretrained backbone and save everything later steps need."""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import config as C, metrics, modeling
from .data import XrayDataset


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature scaling (Guo et al. 2017): one number that makes the probabilities honest.
    Chosen on validation data by minimising the log loss."""
    grid = np.exp(np.linspace(np.log(0.25), np.log(8.0), 200))
    y = np.asarray(y, dtype=np.float64)

    def nll(t):
        p = np.clip(modeling.sigmoid(logits, t), 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    return float(grid[np.argmin([nll(t) for t in grid])])


def train_model(train_sets, val_set, out_dir, *, arch=C.DEFAULT_ARCH, epochs=6, lr=3e-4, weight_decay=1e-4,
                batch_size=32, threads=4, workers=2, seed=C.SEED, init_from=None, extra_config=None, log=print):
    """train_sets: list of (X, indices, labels). val_set: one (X, indices, labels).

    Keeps the epoch with the best validation AUROC. Picks the decision threshold
    (95% sensitivity) and the temperature on the validation set, never on test data.
    -> (model, config dict)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(threads)
    out_dir = Path(out_dir)

    if init_from:
        model, _ = modeling.load_checkpoint(init_from)
    else:
        model = modeling.build_model(arch, pretrained=True)
    model = model.to(memory_format=torch.channels_last)

    train_ds = torch.utils.data.ConcatDataset([XrayDataset(X, i, y, train=True) for X, i, y in train_sets])
    loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                                         num_workers=workers, persistent_workers=workers > 0)
    Xv, iv, yv = val_set
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total = epochs * len(loader)
    warm = len(loader) // 2
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, total - warm))))
    loss_fn = nn.BCEWithLogitsLoss()

    history, best_auc, best_state = [], -1.0, None
    step, t0 = 0, time.time()
    for ep in range(epochs):
        model.train()
        run, seen = 0.0, 0
        for x, y, _ in loader:
            x = x.to(memory_format=torch.channels_last)
            loss = loss_fn(model(x).squeeze(1), y.float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run += loss.item() * len(y); seen += len(y); step += 1
            if step == 20:
                log(f"  speed: {20 * batch_size / (time.time() - t0):.1f} img/s, {len(loader)} steps per epoch")
        val_logits = modeling.predict(model, Xv, iv)
        auc = metrics.auroc(yv, val_logits)
        history.append(dict(epoch=ep + 1, train_loss=run / seen, val_auroc=auc, minutes=(time.time() - t0) / 60))
        log(f"  epoch {ep + 1}/{epochs}  train loss {run / seen:.4f}  val AUROC {auc:.4f}  ({(time.time() - t0) / 60:.1f} min)")
        if auc > best_auc:
            best_auc, best_state = auc, {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    val_logits = modeling.predict(model, Xv, iv)
    temperature = fit_temperature(val_logits, yv)
    val_p = modeling.sigmoid(val_logits, temperature)
    threshold = metrics.threshold_for_sensitivity(yv, val_p, 0.95)
    cfg = dict(arch=arch, temperature=temperature, threshold=threshold,
               threshold_rule="95% sensitivity on the validation set",
               n_train=len(train_ds), epochs=epochs, lr=lr, batch_size=batch_size, seed=seed,
               best_val_auroc=best_auc, history=history, train_minutes=(time.time() - t0) / 60)
    cfg.update(extra_config or {})
    modeling.save_checkpoint(model, out_dir, cfg)
    return model, cfg


def export_predictions(model, out_dir, datasets: dict, log=print) -> None:
    """Run the final model over every image of every dataset and store logits and embeddings,
    so that all later analysis (subgroups, OOD warning, learning curves) is fast and never
    needs the network again. Files: <out_dir>/preds_<name>.npz with arrays `logits`, `emb`."""
    out_dir = Path(out_dir)
    for name, X in datasets.items():
        t = time.time()
        logits, emb = modeling.predict(model, X, return_embeddings=True)
        np.savez(out_dir / f"preds_{name}.npz", logits=logits.astype(np.float32), emb=emb.astype(np.float32))
        log(f"  predictions for {name}: {len(logits)} images in {time.time() - t:.0f}s")


def headline(model_dir, kermany_meta, rsna_meta) -> dict:
    """The first table: same model, same threshold, children vs adults."""
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text())
    T, thr = cfg["temperature"], cfg["threshold"]
    pk = modeling.sigmoid(np.load(model_dir / "preds_kermany.npz")["logits"], T)
    pr = modeling.sigmoid(np.load(model_dir / "preds_rsna.npz")["logits"], T)
    k, r = kermany_meta, rsna_meta
    out = {}
    m = (k.split == "test").values
    out["children_test (Kermany, pneumonia vs normal)"] = metrics.summarize(k.label[m], pk[m], thr)
    adults = ((r.split == "test") & r.is_adult).values
    m = adults & (r.cls != "No Lung Opacity / Not Normal").values
    out["adults_test (RSNA, lung opacity vs normal)"] = metrics.summarize(r.label[m], pr[m], thr)
    out["adults_test (RSNA, lung opacity vs everything else)"] = metrics.summarize(r.label[adults], pr[adults], thr)
    return out
