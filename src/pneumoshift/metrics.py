"""Evaluation helpers. Every number in the write-up should come from these functions."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def auroc(y, p) -> float:
    y = np.asarray(y)
    return float("nan") if len(np.unique(y)) < 2 else float(roc_auc_score(y, p))


def threshold_for_sensitivity(y, p, target: float = 0.95) -> float:
    """Highest threshold that still catches at least `target` of the true pneumonia cases.
    Choose it on VALIDATION data, then keep it fixed on the test data."""
    pos = np.sort(np.asarray(p)[np.asarray(y) == 1])
    k = int(np.floor((1 - target) * len(pos)))
    return float(pos[k])


def confusion(y, p, thr: float) -> dict:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    div = lambda a, b: float(a / b) if b else float("nan")
    return dict(tp=tp, fp=fp, tn=tn, fn=fn,
                sensitivity=div(tp, tp + fn), specificity=div(tn, tn + fp),
                ppv=div(tp, tp + fp), npv=div(tn, tn + fn),
                accuracy=div(tp + tn, len(y)), flagged_rate=div(tp + fp, len(y)))


def bootstrap_ci(fn, y, p, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """95% confidence interval for fn(y, p) by resampling the test cases."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        v = fn(y[i], p[i])
        if not np.isnan(v):
            vals.append(v)
    return (float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2))))


def ece(y, p, n_bins: int = 10) -> float:
    """Expected calibration error: when the model says 80%, is it right 80% of the time?"""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def summarize(y, p, thr: float, n_boot: int = 1000, seed: int = 0) -> dict:
    """All headline numbers for one test group, with the majority-class baseline next to them."""
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    prev = float(y.mean())
    out = dict(n=int(len(y)), n_pos=int(y.sum()), prevalence=prev,
               majority_baseline_accuracy=max(prev, 1 - prev),
               auroc=auroc(y, p), auprc=float(average_precision_score(y, p)) if 0 < prev < 1 else float("nan"),
               threshold=float(thr), ece=ece(y, p))
    out.update(confusion(y, p, thr))
    out["balanced_accuracy"] = (out["sensitivity"] + out["specificity"]) / 2
    if n_boot and 0 < prev < 1:
        out["auroc_ci"] = bootstrap_ci(auroc, y, p, n_boot, seed)
        out["sensitivity_ci"] = bootstrap_ci(lambda a, b: confusion(a, b, thr)["sensitivity"], y, p, n_boot, seed)
        out["specificity_ci"] = bootstrap_ci(lambda a, b: confusion(a, b, thr)["specificity"], y, p, n_boot, seed)
    return out
