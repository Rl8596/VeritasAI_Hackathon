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

Outputs: results/baselines.json, results/figures/baseline_*.png, and per-image probabilities plus cached
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

from pneumoshift import config as C, data, metrics, modeling, plotting  # noqa: E402
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
    """512-d ResNet-18 features for X[idx], cached on disk."""
    if cache_file.exists():
        z = np.load(cache_file)
        if np.array_equal(z["idx"], idx):
            log(f"   loaded cached embeddings {cache_file}")
            return z["emb"]
    parts, t = [], time.time()
    for s in range(0, len(idx), 1024):
        parts.append(modeling.predict(model, X, idx[s:s + 1024], batch_size=64, return_embeddings=True)[1])
        done = min(s + 1024, len(idx))
        log(f"   embedded {done}/{len(idx)} images ({done / (time.time() - t):.1f} img/s)")
    emb = np.concatenate(parts).astype(np.float32)
    np.savez(cache_file, idx=idx, emb=emb)
    return emb


SCRATCH.mkdir(parents=True, exist_ok=True)
C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
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

# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
import matplotlib.pyplot as plt  # noqa: E402

plotting.use_style()
ROWS = [("majority_class", "Always say\n\"pneumonia\""),
        ("pixels_32x32_logreg", "Logistic regression\non 32×32 pixels"),
        ("frozen_imagenet_resnet18_logreg", "ImageNet ResNet-18,\nnot fine-tuned,\n+ logistic regression")]
GROUPS = [(KEY_CHILD, "Children, test set (pneumonia vs normal)", plotting.CHILDREN, plotting.CHILDREN, None),
          (KEY_ADULT_NORMAL, "Adults, test set (lung opacity vs normal)", plotting.ADULTS, plotting.ADULTS, None),
          (KEY_ADULT_ALL, "Adults, test set (lung opacity vs everything else)", plotting.ADULTS, plotting.SURFACE, "////")]


def r(name, key):
    return baselines[name]["results"][key]


# Figure 1: AUROC of every baseline in every test group
fig, ax = plt.subplots(figsize=(10, 5.4))
for i, (name, _) in enumerate(ROWS):
    for j, (key, label, edge, face, _) in enumerate(GROUPS):
        v = r(name, key)
        y = i + (j - 1) * 0.24
        lo, hi = v.get("auroc_ci", (v["auroc"], v["auroc"]))
        ax.plot([lo, hi], [y, y], color=edge, lw=2, solid_capstyle="round", zorder=2)
        ax.plot(v["auroc"], y, "o", ms=11, mfc=face, mec=edge, mew=2.2, zorder=3, label=label if i == 0 else None)
        ax.annotate(f"{v['auroc']:.2f}", (hi, y), xytext=(10, 0), textcoords="offset points", va="center",
                    fontsize=12, color=plotting.TEXT)
ax.axvline(0.5, color=plotting.NEUTRAL, lw=1.5, zorder=1)
ax.annotate("coin flip", (0.5, -0.52), xytext=(6, 0), textcoords="offset points", va="center", fontsize=11,
            color=plotting.TEXT_2)
ax.set_yticks(range(len(ROWS)), [lab for _, lab in ROWS])
ax.set_ylim(len(ROWS) - 0.45, -0.7)
ax.set_xlim(0.38, 1.06)
ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.grid(axis="y", visible=False)
ax.tick_params(axis="y", length=0, labelsize=12)
ax.set_xlabel("AUROC: how well the score ranks sick above healthy (1.0 = perfect, 0.5 = coin flip)")
best = max(("pixels_32x32_logreg", "frozen_imagenet_resnet18_logreg"), key=lambda n: r(n, KEY_CHILD)["auroc"])
ax.set_title(f"Simple baselines trained on children: AUROC {r(best, KEY_CHILD)['auroc']:.2f} on children, "
             f"{r(best, KEY_ADULT_NORMAL)['auroc']:.2f} and {r(best, KEY_ADULT_ALL)['auroc']:.2f} on adults", pad=14)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, handletextpad=0.4)
plotting.finish(fig, C.FIGURES_DIR / "baseline_auroc.png",
                note=f"Trained on Kermany train ({k_train.sum()} children's X-rays) only. Test sets: Kermany test n={k_test.sum()}, "
                     f"RSNA adults n={r(best, KEY_ADULT_NORMAL)['n']} / {r(best, KEY_ADULT_ALL)['n']}. "
                     "Lines: 95% bootstrap intervals.")

# Figure 2: what the fixed threshold does in each group (sensitivity and specificity)
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
for ax, (name, title) in zip(axes, [("pixels_32x32_logreg", "Logistic regression on 32×32 pixels"),
                                    ("frozen_imagenet_resnet18_logreg", "ImageNet ResNet-18 (not fine-tuned) + logistic regression")]):
    for m, metric in enumerate(("sensitivity", "specificity")):
        for j, (key, label, edge, face, hatch) in enumerate(GROUPS):
            v = r(name, key)
            x = m + (j - 1) * 0.27
            ax.bar(x, 100 * v[metric], width=0.23, color=face, edgecolor=edge, linewidth=1.6, hatch=hatch,
                   label=label if m == 0 else None, zorder=3)
            ax.annotate(f"{100 * v[metric]:.0f}%", (x, 100 * v[metric]), xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=11, color=plotting.TEXT)
    ax.set_xticks([0, 1], ["Sick patients flagged\n(sensitivity)", "Healthy patients cleared\n(specificity)"])
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="x", visible=False)
    ax.set_title(title, fontsize=12, fontweight="normal", color=plotting.TEXT_2)
axes[0].legend(loc="upper center", bbox_to_anchor=(1.0, -0.2), ncol=1, handletextpad=0.5)
b3 = "frozen_imagenet_resnet18_logreg"
fig.suptitle(f"A threshold tuned on children does not carry over: the best baseline catches "
             f"{100 * r(b3, KEY_CHILD)['sensitivity']:.0f}% of sick children\nbut "
             f"{100 * r(b3, KEY_ADULT_ALL)['sensitivity']:.0f}% of sick adults, and clears "
             f"{100 * r(b3, KEY_CHILD)['specificity']:.0f}% vs {100 * r(b3, KEY_ADULT_ALL)['specificity']:.0f}% of the healthy",
             x=0.01, ha="left", fontsize=14, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.98))
plotting.finish(fig, C.FIGURES_DIR / "baseline_sensitivity_specificity.png",
                note="Decision threshold chosen once, on Kermany validation images, to catch 95% of pneumonia cases; never re-tuned. "
                     "Adults: RSNA test set.")
log("wrote figures baseline_auroc.png, baseline_sensitivity_specificity.png")
