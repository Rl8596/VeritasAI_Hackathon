"""Step 11 (exploratory): would a network pretrained on chest X-rays from OTHER hospitals do better?

    pip install torchxrayvision
    python scripts/11_xray_pretrained_check.py

Why this exists. The first v2 result (validation data, 14 Nov) showed that 15,142 adult X-rays give the same
adult score as 6,000: with our recipe, more data of the same kind does not help. This script tests the other
obvious lever, a better starting point. It was added AFTER that result was seen, it is not part of
docs/V2_PROTOCOL.md, and its rows are labelled "exploratory" wherever they appear.

What is fixed here, before any number was computed, so that nothing can be picked afterwards:

  - One model only: the DenseNet-121 of the TorchXRayVision library that was trained on CheXpert
    (Stanford Hospital). Our adult X-rays come from the NIH Clinical Center, so models trained on NIH or
    RSNA data, or on "all" datasets, have seen our test X-rays and are NOT used.
  - Input: the stored 256 px picture, resized to 224 px (the size and framing these weights were trained
    with) and scaled to the value range the library expects. No fine-tuning: the network stays as published.
  - Two ways of using it, both reported:
      1. as it is: its own output for "Lung Opacity" (it never sees one of our X-rays in training)
      2. its 1,024 internal numbers + logistic regression trained on our adult TRAINING X-rays; the
         regularisation strength is chosen on adult VALIDATION X-rays (same grid as the ImageNet baseline)

Output: results/exploratory/xray_pretrained_chex.npz (one score per adult-hospital X-ray for each way) and
a .json beside it. scripts/10_v2_report.py picks the file up and adds the two rows to its tables.
Runs on a CPU in about 20 minutes; the features are cached in the scratch folder.
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pneumoshift import config as C, data, metrics  # noqa: E402

WEIGHTS = "densenet121-res224-chex"
OUT = C.RESULTS_DIR / "exploratory"
SCRATCH = Path(os.environ.get("PNEUMOSHIFT_SCRATCH", Path(tempfile.gettempdir()) / "pneumoshift_scratch"))

if __name__ == "__main__":
    import torchxrayvision as xrv
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    Xr, mr = data.load_rsna()
    y = mr.label.values.astype(int)
    adult = mr.is_adult.values
    train = (mr.split == "adapt").values & adult
    val = (mr.split == "val").values & adult

    cache = SCRATCH / f"xrv_{WEIGHTS}_rsna.npz"
    if cache.exists() and len(np.load(cache)["features"]) == len(mr):
        z = np.load(cache)
        feats, outs, labels = z["features"], z["outputs"], [str(s) for s in z["labels"]]
    else:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
        model = xrv.models.DenseNet(weights=WEIGHTS).eval()
        labels = list(model.pathologies)
        parts = SCRATCH / f"xrv_{WEIGHTS}_parts"          # one file per 3,200 X-rays, so an interrupted run continues
        parts.mkdir(parents=True, exist_ok=True)
        feats, outs, t0 = [], [], time.time()
        with torch.no_grad():
            for s in range(0, len(Xr), 3200):
                part = parts / f"{s:06d}.npz"
                if not part.exists():
                    f_, o_ = [], []
                    for i in range(s, min(s + 3200, len(Xr)), 64):
                        x = torch.from_numpy(np.ascontiguousarray(Xr[i:i + 64])).float()[:, None]  # [B, 1, 256, 256], 0..255
                        x = F.interpolate(x, size=(224, 224), mode="bilinear", antialias=True, align_corners=False)
                        x = (2 * (x / 255.0) - 1.0) * 1024.0                                       # the library's value range
                        f_.append(model.features2(x).numpy().astype(np.float32))
                        o_.append(model(x).numpy().astype(np.float32))
                    tmp = parts / f"{s:06d}.tmp.npz"
                    np.savez(tmp, features=np.concatenate(f_), outputs=np.concatenate(o_))
                    tmp.replace(part)
                    print(f"  {min(s + 3200, len(Xr)):6d} of {len(Xr)} X-rays ({time.time() - t0:.0f}s)", flush=True)
                z = np.load(part)
                feats.append(z["features"])
                outs.append(z["outputs"])
        feats, outs = np.concatenate(feats), np.concatenate(outs)
        np.savez(cache, features=feats, outputs=outs, labels=np.array(labels))

    as_is = outs[:, labels.index("Lung Opacity")].astype(np.float64)
    as_is_pneumonia = outs[:, labels.index("Pneumonia")].astype(np.float64)

    sc = StandardScaler().fit(feats[train])
    best = None
    for c in (1e-3, 1e-2, 1e-1, 1.0):                      # chosen on adult VALIDATION X-rays
        clf = LogisticRegression(C=c, max_iter=3000).fit(sc.transform(feats[train]), y[train])
        a = metrics.auroc(y[val], clf.decision_function(sc.transform(feats[val])))
        if best is None or a > best[0]:
            best = (a, c, clf)
    frozen = best[2].decision_function(sc.transform(feats))

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "xray_pretrained_chex.npz", as_is_lung_opacity=as_is.astype(np.float32),
                        as_is_pneumonia=as_is_pneumonia.astype(np.float32), frozen_features_logreg=frozen.astype(np.float32))
    info = dict(
        status="exploratory: added on 14 Nov 2026 after the first v2 validation result, not part of docs/V2_PROTOCOL.md",
        model=f"TorchXRayVision DenseNet-121, weights '{WEIGHTS}' (trained on CheXpert, Stanford Hospital; no NIH or RSNA X-ray)",
        input="stored 256 px picture resized to 224 px, values scaled to -1024..1024, no fine-tuning",
        rows=dict(
            xray_pretrained_as_is="the network's own output for 'Lung Opacity', used as published (it never saw one of our X-rays)",
            xray_pretrained_frozen="logistic regression on the network's 1,024 features, trained on the adult training X-rays; "
                                   f"C = {best[1]} chosen on adult validation X-rays"),
        validation=dict(adult_auroc_as_is=metrics.auroc(y[val], as_is[val]), adult_auroc_frozen=best[0],
                        note="validation X-rays only; test results come from scripts/10_v2_report.py"),
        n_features=int(feats.shape[1]), chosen_C=best[1])
    (OUT / "xray_pretrained_chex.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info["validation"], indent=1))
    print(f"wrote {OUT / 'xray_pretrained_chex.npz'}")
