"""Fine-tune the pretrained backbone and save everything later steps need."""
import json
import os
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
                batch_size=32, threads=4, workers=2, seed=C.SEED, init_from=None, extra_config=None, log=print,
                amp=False, require_gpu=False, report_sets=None, select="best"):
    """train_sets: list of (X, indices, labels). val_set: one (X, indices, labels).

    Keeps the epoch with the best AUROC on val_set. Picks the decision threshold
    (95% sensitivity) and the temperature on val_set, never on test data.

    report_sets: optional {name: (X, indices, labels)} of further VALIDATION groups. Their AUROC is
    recorded after every epoch and their results at the final threshold are stored in the config
    under "validation". They are only reported; nothing is chosen on them.
    select: "best" keeps the epoch with the best AUROC on val_set (the first models). "last" keeps the final
    epoch, so that the kept weights do not depend on any validation X-ray. That matters for the question
    "how many local X-rays does a clinic need": a clinic with 300 labelled X-rays has no 2,500 more to pick an epoch.
    amp: mixed precision on the GPU (about twice as fast, half the memory). Off by default, so a
    run without the flag follows the same recipe as the first models.
    -> (model, config dict)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(threads)
    out_dir = Path(out_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"   # GPU if there is one, CPU elsewhere
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        log(f"  device: {torch.cuda.get_device_name(0)} (mixed precision {'on' if amp else 'off'})")
    else:
        if require_gpu:
            raise RuntimeError("No GPU found, but --require-gpu was given. Is PyTorch installed with CUDA support? "
                               "(python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\")")
        log("  device: CPU (no GPU found, this takes 20 to 60 minutes per model)")
    amp = bool(amp) and device == "cuda"

    if init_from:
        model, _ = modeling.load_checkpoint(init_from)
    else:
        model = modeling.build_model(arch, pretrained=True)
    model = model.to(device).to(memory_format=torch.channels_last)

    train_ds = torch.utils.data.ConcatDataset([XrayDataset(X, i, y, train=True) for X, i, y in train_sets])
    # PNEUMOSHIFT_MP=spawn starts the worker processes the way Windows does (to test that on Linux)
    mp_context = (os.environ.get("PNEUMOSHIFT_MP") or None) if workers > 0 else None
    loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                                         num_workers=workers, persistent_workers=workers > 0,
                                         pin_memory=device == "cuda", multiprocessing_context=mp_context)
    if len(loader) == 0:
        raise ValueError(f"only {len(train_ds)} training images, fewer than one batch of {batch_size}")
    Xv, iv, yv = val_set
    report_sets = dict(report_sets or {})
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total = epochs * len(loader)
    warm = max(1, len(loader) // 2)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, total - warm))))
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    history, best_auc, best_state, best_epoch = [], -1.0, None, 0
    step, t0, t_speed = 0, time.time(), None
    for ep in range(epochs):
        model.train()
        run, seen = 0.0, 0
        for x, y, _ in loader:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.float().to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                logits = model(x).squeeze(1)
            loss = loss_fn(logits.float(), y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"the training loss became {loss.item()} at step {step}: stopping instead of saving a broken model")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item() * len(y); seen += len(y); step += 1
            if step == 10:                       # the first steps include starting the worker processes
                t_speed = time.time()
            if step == 60 and t_speed is not None:
                mem = f", GPU memory peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB" if device == "cuda" else ""
                log(f"  speed: {50 * batch_size / (time.time() - t_speed):.0f} img/s, {len(loader)} steps per epoch{mem}")
        val_logits = modeling.predict(model, Xv, iv, device=device)
        auc = metrics.auroc(yv, val_logits)
        row = dict(epoch=ep + 1, train_loss=run / seen, val_auroc=auc, minutes=(time.time() - t0) / 60)
        for name, (Xg, ig, yg) in report_sets.items():
            row[f"val_auroc_{name}"] = metrics.auroc(yg, modeling.predict(model, Xg, ig, device=device))
        history.append(row)
        extra = "".join(f"  {name} {row[f'val_auroc_{name}']:.4f}" for name in report_sets)
        log(f"  epoch {ep + 1}/{epochs}  train loss {run / seen:.4f}  val AUROC {auc:.4f}{extra}  ({(time.time() - t0) / 60:.1f} min)")
        if (select == "last" and ep == epochs - 1) or (select != "last" and auc > best_auc):
            best_auc, best_state = auc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = ep + 1

    if best_state is None or not np.isfinite(best_auc):
        raise RuntimeError("the validation AUROC could not be computed (does the validation set hold both classes?)")
    model.load_state_dict(best_state)
    model.eval()
    val_logits = modeling.predict(model, Xv, iv, device=device)
    temperature = fit_temperature(val_logits, yv)
    val_p = modeling.sigmoid(val_logits, temperature)
    threshold = metrics.threshold_for_sensitivity(yv, val_p, 0.95)
    validation = {}
    for name, (Xg, ig, yg) in report_sets.items():
        pg = modeling.sigmoid(modeling.predict(model, Xg, ig, device=device), temperature)
        g = metrics.summarize(yg, pg, threshold, n_boot=0)
        validation[name] = {k: g[k] for k in ("n", "n_pos", "auroc", "sensitivity", "specificity", "accuracy")}
    cfg = dict(arch=arch, temperature=temperature, threshold=threshold,
               threshold_rule="95% sensitivity on the validation set",
               n_train=len(train_ds), epochs=epochs, lr=lr, batch_size=batch_size, seed=seed,
               best_val_auroc=best_auc, best_epoch=best_epoch, epoch_rule=select, history=history, train_minutes=(time.time() - t0) / 60,
               mixed_precision=amp, device=torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
               torch_version=torch.__version__, validation=validation)
    cfg.update(extra_config or {})
    modeling.save_checkpoint(model, out_dir, cfg)
    return model, cfg


def export_predictions(model, out_dir, datasets: dict, log=print) -> None:
    """Run the final model over every image of every dataset and store logits and embeddings,
    so that all later analysis (subgroups, OOD warning, learning curves) is fast and never
    needs the network again. Files: <out_dir>/preds_<name>.npz with arrays `logits`, `emb`."""
    out_dir = Path(out_dir)
    device = next(model.parameters()).device
    for name, X in datasets.items():
        t = time.time()
        logits, emb = modeling.predict(model, X, return_embeddings=True, device=device)
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
