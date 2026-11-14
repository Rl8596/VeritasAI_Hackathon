"""Step 7: how many adult X-rays does it take to repair the children-only model?

    python scripts/07_finetune_curve.py

Compares two ways of using n labelled adult X-rays:
  - retrain the whole network on children + n adults (one model per n, trained with 02_train.py)
  - retrain only the last layer (numbers from 05_shift_analysis.py --model kids_only)
Reads results/headline_<model>.json, writes results/finetune_curve.json and a figure.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pneumoshift import config as C, plotting  # noqa: E402

FULL = [("kids_only", 0), ("kids_plus_300", 300), ("kids_plus_1000", 1000), ("kids_plus_adults", 6000)]
ADULT, ADULT_EASY, CHILD = ("adults_test (RSNA, lung opacity vs everything else)",
                            "adults_test (RSNA, lung opacity vs normal)",
                            "children_test (Kermany, pneumonia vs normal)")

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rows = []
    for name, n in FULL + [("adults_only", 6000)]:
        f = C.RESULTS_DIR / f"headline_{name}.json"
        if not f.exists():
            print("missing", f.name); continue
        r = json.loads(f.read_text())["results"]
        rows.append(dict(model=name, n_adults=n, threshold=json.loads(f.read_text())["threshold"],
                         **{f"{k}_{g}": r[key][k] for g, key in (("adults", ADULT), ("adults_vs_normal", ADULT_EASY), ("children", CHILD))
                            for k in ("auroc", "sensitivity", "specificity", "accuracy", "majority_baseline_accuracy")}))
    last = json.loads((C.RESULTS_DIR / "shift_kids_only.json").read_text())["learning_curve"]
    last = [r for r in last if r["strategy"] == "children + n adults"]
    ns = sorted({r["n_adults"] for r in last})
    last_layer = [dict(n_adults=n, auroc_adults=float(np.mean([r["auroc_adults"] for r in last if r["n_adults"] == n])),
                       auroc_children=float(np.mean([r["auroc_children"] for r in last if r["n_adults"] == n]))) for n in ns]
    (C.RESULTS_DIR / "finetune_curve.json").write_text(json.dumps(dict(whole_network=rows, last_layer_only=last_layer), indent=2))
    for r in rows:
        print(f"{r['model']:18s} n={r['n_adults']:5d} thr {r['threshold']:.3f} | adults AUROC {r['auroc_adults']:.3f} sens {r['sensitivity_adults']:.2f} "
              f"spec {r['specificity_adults']:.2f} acc {r['accuracy_adults']:.3f} (base {r['majority_baseline_accuracy_adults']:.3f}) | "
              f"children AUROC {r['auroc_children']:.3f} sens {r['sensitivity_children']:.2f} spec {r['specificity_children']:.2f}")

    plotting.use_style()
    full = [r for r in rows if r["model"] != "adults_only"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3), sharey=True)
    for ax, key, title in ((axes[0], "auroc_adults", "Adult test X-rays (other hospital)"),
                           (axes[1], "auroc_children", "Children test X-rays (training hospital)")):
        ax.plot([r["n_adults"] for r in last_layer], [r[key] for r in last_layer], color=plotting.NEUTRAL, marker="o", label="Retrain last layer only")
        ax.plot([r["n_adults"] for r in full], [r[key] for r in full], color=plotting.THIRD, marker="o", label="Retrain whole network")
        for r in full:
            ax.annotate(f"{r[key]:.2f}", (r["n_adults"], r[key]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=10)
        only = [r for r in rows if r["model"] == "adults_only"]
        if only:
            ax.axhline(only[0][key], color=plotting.ADULTS, linestyle="--", linewidth=1.5, label="Adults-only model (6,000 X-rays)")
        ax.set_xscale("symlog", linthresh=100)
        ax.set_xticks([0, 100, 300, 1000, 3000, 10000], ["0", "100", "300", "1,000", "3,000", "10,000"])
        ax.set_xlim(-10, 20000)
        ax.set_xlabel("Adult X-rays added to training")
        ax.set_title(title, fontsize=12)
    axes[0].set_ylabel("AUROC (0.5 = guessing, 1 = perfect)")
    axes[0].legend(loc="lower right")
    fig.suptitle("How many adult X-rays does it take to repair the children-only model?", x=0.01, ha="left", fontweight="bold", fontsize=14)
    fig.tight_layout()
    plotting.finish(fig, C.FIGURES_DIR / "finetune_curve.png",
                    note="Adults: lung opacity vs everything else, held-out patients. Last-layer points are the mean of 5 random draws.")
