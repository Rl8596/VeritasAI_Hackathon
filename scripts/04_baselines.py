"""Step 4: simple baselines that the fine-tuned model has to beat.

    python scripts/04_baselines.py

Three baselines, all following the project's evaluation protocol:
  * everything that is learned (weights, scalers) uses Kermany `split == "train"` only;
  * every choice (regularisation strength, decision threshold) uses Kermany `split == "val"`;
  * the threshold is the one that catches 95% of pneumonia cases on Kermany val, then stays fixed;
  * test data (Kermany test, RSNA test adults) is only ever scored, never used to choose anything.

  1. majority class            always answers "pneumonia" (the most common class in Kermany train)
  2. pixels + logistic reg.    the 224 px centre crop the network sees, shrunk to 32x32 = 1024 numbers
  3. frozen ImageNet features  the pretrained ResNet-18 WITHOUT any fine-tuning (512 numbers per image)
                               + logistic regression

Outputs: results/baselines.json, and per-image probabilities plus cached
embeddings in the scratch folder (PNEUMOSHIFT_SCRATCH, default: <system temp dir>/pneumoshift_scratch).
One CPU thread by default (set OMP_NUM_THREADS to use more); embedding ~13,500 images takes about
10 minutes on one thread the first time, a few seconds afterwards thanks to the cache.
"""
import os

THREADS = int(os.environ.get("OMP_NUM_THREADS", "1"))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = str(THREADS)

import json  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pneumoshift import config as C, data, metrics, modeling  # noqa: E402
from pneumoshift.preprocess import center_crop  # noqa: E402

torch.set_num_threads(THREADS)

SCRATCH = Path(os.environ.get("PNEUMOSHIFT_SCRATCH", Path(tempfile.gettempdir()) / "pneumoshift_scratch"))
RSNA_MAX = None          # None = every adult test image; an integer = fixed random subset of that many
TARGET_SENSITIVITY = 0.95
PIXEL_C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
EMB_C_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0]

KEY_CHILD = "children_test (Kermany, pneumonia vs normal)"
KEY_ADULT_NORMAL = "adults_test (RSNA, lung opacity vs normal)"
KEY_ADULT_ALL = "adults_test (RSNA, lung opacity vs everything else)"
KEY_VAL = "validation (Kermany val, used for choosing C and the threshold, not a result)"

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def clean(o):
    """Make numpy types and NaN safe for strict JSON."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.integer, np.bool_)):
        return o.item()
    return o


# --------------------------------------------------------------------------
# Data and the three test groups (same selection as src/pneumoshift/train.py::headline)
# --------------------------------------------------------------------------
Xk, mk = data.load_kermany()
Xr, mr = data.load_rsna()
yk, yr = mk.label.values, mr.label.values
k_train, k_val, k_test = [(mk.split == s).values for s in ("train", "val", "test")]
adults_test = ((mr.split == "test") & mr.is_adult).values
adults_test_vs_normal = adults_test & (mr.cls != "No Lung Opacity / Not Normal").values

rsna_idx = mr.idx.values[adults_test]                 # the only RSNA images this script touches
if RSNA_MAX and RSNA_MAX < len(rsna_idx):
    rsna_idx = np.sort(np.random.default_rng(C.SEED).choice(rsna_idx, RSNA_MAX, replace=False))
used = np.zeros(len(mr), bool)
used[rsna_idx] = True
log(f"Kermany: {k_train.sum()} train / {k_val.sum()} val / {k_test.sum()} test images; "
    f"RSNA adult test: {len(rsna_idx)} of {adults_test.sum()} images used")


def evaluate(name, description, pk, pr, extra=None):
    """pk: probability for every Kermany image; pr: probability for the RSNA images in rsna_idx."""
    thr = metrics.threshold_for_sensitivity(yk[k_val], pk[k_val], TARGET_SENSITIVITY)
    pr_full = np.full(len(mr), np.nan)
    pr_full[rsna_idx] = pr
    m_all, m_norm = adults_test & used, adults_test_vs_normal & used
    res = {
        KEY_CHILD: metrics.summarize(yk[k_test], pk[k_test], thr),
        KEY_ADULT_NORMAL: metrics.summarize(yr[m_norm], pr_full[m_norm], thr),
        KEY_ADULT_ALL: metrics.summarize(yr[m_all], pr_full[m_all], thr),
    }
    out = dict(model=name, description=description, threshold=thr,
               threshold_rule=f"{TARGET_SENSITIVITY:.0%} sensitivity on Kermany val, then fixed")
    out.update(extra or {})
    out["results"] = res
    out[KEY_VAL] = metrics.summarize(yk[k_val], pk[k_val], thr, n_boot=0)
    log(f"{name}: threshold {thr:.4f}")
    for k, v in res.items():
        log(f"   {k:52s} n={v['n']:5d}  AUROC {v['auroc']:.3f}  sens {v['sensitivity']:.3f}  spec {v['specificity']:.3f}"
            f"  acc {v['accuracy']:.3f}  (majority {v['majority_baseline_accuracy']:.3f})")
    return out


def fit_logreg(F_train, y_train, F_val, y_val, grid):
    """Standardise on train, fit one model per C on train, keep the C with the best val AUROC."""
    scaler = StandardScaler().fit(F_train)
    A, V = scaler.transform(F_train), scaler.transform(F_val)
    val_auroc, best = {}, None
    for c in grid:
        clf = LogisticRegression(C=c, max_iter=5000).fit(A, y_train)
        val_auroc[c] = metrics.auroc(y_val, clf.predict_proba(V)[:, 1])
        if best is None or val_auroc[c] > best[0]:
            best = (val_auroc[c], c, clf)
    return scaler, best[2], best[1], val_auroc


def pixels_32(X, idx):
    """224 px centre crop (what the network sees) -> 32x32 block means -> 1024 values in [0, 1]."""
    out = np.zeros((len(idx), 32 * 32), np.float32)
    for s in range(0, len(idx), 512):
        a = center_crop(np.asarray(X[idx[s:s + 512]]), C.IMG_SIZE).astype(np.float32)
        out[s:s + len(a)] = a.reshape(len(a), 32, 7, 32, 7).mean((2, 4)).reshape(len(a), -1) / 255.0
    return out


def embeddings(model, X, idx, cache_file):
    """512-d ResNet-18 features for X[idx]. Cached on disk per image, so only new images are computed."""
    have_idx, have = np.zeros(0, np.int64), None
    if cache_file.exists():
        z = np.load(cache_file)
        have_idx, have = z["idx"], z["emb"]
    missing = np.setdiff1d(idx, have_idx)
    log(f"   {cache_file.name}: {len(idx) - len(missing)} of {len(idx)} embeddings cached, computing {len(missing)}")
    if len(missing):
        parts, t = [], time.time()
        for s in range(0, len(missing), 1024):
            parts.append(modeling.predict(model, X, missing[s:s + 1024], batch_size=64, return_embeddings=True)[1])
            done = min(s + 1024, len(missing))
            log(f"   embedded {done}/{len(missing)} images ({done / (time.time() - t):.1f} img/s)")
        new = np.concatenate(parts).astype(np.float32)
        have = new if have is None else np.concatenate([have, new])
        have_idx = np.concatenate([have_idx, missing])
        order = np.argsort(have_idx)
        have_idx, have = have_idx[order], have[order]
        np.savez(cache_file, idx=have_idx, emb=have)
    return have[np.searchsorted(have_idx, idx)]


SCRATCH.mkdir(parents=True, exist_ok=True)
baselines = {}
all_k = np.arange(len(mk))

# --------------------------------------------------------------------------
# 1. Majority class: always "pneumonia" (constant score = share of pneumonia in Kermany train)
# --------------------------------------------------------------------------
prev = float(yk[k_train].mean())
baselines["majority_class"] = evaluate(
    "majority_class", "Always answers 'pneumonia', the most common class in Kermany train. "
    "The constant score is the share of pneumonia images in Kermany train.",
    np.full(len(mk), prev), np.full(len(rsna_idx), prev), dict(train_prevalence=prev))

# --------------------------------------------------------------------------
# 2. Logistic regression on 32x32 pixels
# --------------------------------------------------------------------------
Pk, Pr = pixels_32(Xk, all_k), pixels_32(Xr, rsna_idx)
scaler, clf, best_c, val_auroc = fit_logreg(Pk[k_train], yk[k_train], Pk[k_val], yk[k_val], PIXEL_C_GRID)
pk2, pr2 = clf.predict_proba(scaler.transform(Pk))[:, 1], clf.predict_proba(scaler.transform(Pr))[:, 1]
baselines["pixels_32x32_logreg"] = evaluate(
    "pixels_32x32_logreg", "Logistic regression on the 224 px centre crop shrunk to 32x32 pixels "
    "(standardised with Kermany train statistics, C chosen by AUROC on Kermany val).",
    pk2, pr2, dict(n_features=Pk.shape[1], chosen_C=best_c, val_auroc_by_C=val_auroc))

# --------------------------------------------------------------------------
# 3. Frozen ImageNet ResNet-18 features + logistic regression (no fine-tuning)
# --------------------------------------------------------------------------
log(f"building {C.DEFAULT_ARCH} with ImageNet weights (not fine-tuned)")
net = modeling.build_model(pretrained=True).to(memory_format=torch.channels_last).eval()
tag = C.DEFAULT_ARCH.replace(".", "_")
Ek = embeddings(net, Xk, all_k, SCRATCH / f"emb_kermany_{tag}.npz")
Er = embeddings(net, Xr, rsna_idx, SCRATCH / f"emb_rsna_adult_test_{tag}.npz")
scaler, clf, best_c, val_auroc = fit_logreg(Ek[k_train], yk[k_train], Ek[k_val], yk[k_val], EMB_C_GRID)
pk3, pr3 = clf.predict_proba(scaler.transform(Ek))[:, 1], clf.predict_proba(scaler.transform(Er))[:, 1]
baselines["frozen_imagenet_resnet18_logreg"] = evaluate(
    "frozen_imagenet_resnet18_logreg", f"Logistic regression on the {Ek.shape[1]} pooled features of an "
    f"ImageNet-pretrained {C.DEFAULT_ARCH} that was never trained on X-rays "
    "(standardised with Kermany train statistics, C chosen by AUROC on Kermany val).",
    pk3, pr3, dict(n_features=Ek.shape[1], chosen_C=best_c, val_auroc_by_C=val_auroc))

# --------------------------------------------------------------------------
# Save numbers and per-image probabilities
# --------------------------------------------------------------------------
prob_files = {}
for name, pk, pr in (("pixels_32x32_logreg", pk2, pr2), ("frozen_imagenet_resnet18_logreg", pk3, pr3)):
    f = SCRATCH / f"baseline_probs_{name}.npz"
    np.savez(f, kermany_idx=all_k, kermany_prob=pk.astype(np.float32), rsna_idx=rsna_idx,
             rsna_prob=pr.astype(np.float32), threshold=baselines[name]["threshold"])
    prob_files[name] = str(f)

out = dict(
    protocol=dict(
        learned_on="Kermany split == 'train' only (weights and scalers)",
        chosen_on="Kermany split == 'val' (regularisation strength C and the decision threshold)",
        threshold_rule=f"metrics.threshold_for_sensitivity(y_val, p_val, {TARGET_SENSITIVITY}), fixed for all test groups",
        test_groups={KEY_CHILD: "mk.split == 'test'",
                     KEY_ADULT_NORMAL: "(mr.split == 'test') & mr.is_adult & (mr.cls != 'No Lung Opacity / Not Normal')",
                     KEY_ADULT_ALL: "(mr.split == 'test') & mr.is_adult"},
        note="No baseline ever sees an RSNA image during fitting or tuning. Confidence intervals are the "
             "image-level bootstrap from metrics.summarize."),
    data=dict(kermany_train=int(k_train.sum()), kermany_val=int(k_val.sum()), kermany_test=int(k_test.sum()),
              rsna_adult_test_total=int(adults_test.sum()), rsna_adult_test_used=int(len(rsna_idx)),
              rsna_subsampled=bool(len(rsna_idx) < adults_test.sum())),
    baselines=baselines,
    per_image_probability_files=prob_files,
)
(C.RESULTS_DIR / "baselines.json").write_text(json.dumps(clean(out), indent=2))
log(f"wrote {C.RESULTS_DIR / 'baselines.json'}")
