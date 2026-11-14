"""Step 5: the reality check. Same model, same threshold: children vs adults.

    python scripts/05_shift_analysis.py --model kids_only

Needs artifacts/<model>/ from step 2 (weights, config, predictions for every image).
Writes results/shift_<model>.json, figures results/figures/shift_<model>_*.png and
the out-of-distribution statistics artifacts/<model>/ood.npz.

Nothing in here is tuned on test data: the threshold and temperature come from the
model's config (chosen on validation data in step 2), the OOD warning is fitted on
training images and calibrated on validation images.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pneumoshift import config as C, data, metrics, modeling, plotting  # noqa: E402
from pneumoshift.ood import MahalanobisOOD  # noqa: E402
from pneumoshift.train import headline  # noqa: E402

AGE_BANDS = [("age 1-5", 0, 5), ("age 6-17", 6, 17), ("age 18-39", 18, 39), ("age 40-59", 40, 59), ("age 60+", 60, 100)]


def short(s: dict) -> dict:
    keep = ("n", "n_pos", "prevalence", "auroc", "auroc_ci", "sensitivity", "sensitivity_ci", "specificity",
            "specificity_ci", "accuracy", "majority_baseline_accuracy", "flagged_rate", "ppv", "npv", "ece")
    return {k: s[k] for k in keep if k in s}


def subgroup_table(mr, pr, thr) -> dict:
    """Adult-hospital test set split by age, sex and view. Label: lung opacity vs everything else."""
    test = (mr.split == "test").values
    groups = {name: test & mr.age_valid.values & mr.age.between(lo, hi).values for name, lo, hi in AGE_BANDS}
    groups.update({"female": test & mr.is_adult.values & (mr.sex == "F").values,
                   "male": test & mr.is_adult.values & (mr.sex == "M").values,
                   "view PA (standing, walk-in)": test & mr.is_adult.values & (mr["view"] == "PA").values,
                   "view AP (portable, bedside)": test & mr.is_adult.values & (mr["view"] == "AP").values})
    out = {}
    for name, m in groups.items():
        if m.sum() >= 20 and 0 < mr.label[m].mean() < 1:
            out[name] = short(metrics.summarize(mr.label[m], pr[m], thr, n_boot=500))
    return out


def learning_curve(ek, mk, er, mr, sizes, seeds=5) -> dict:
    """How many labelled adult X-rays does it take to fix the model?

    The network stays frozen. Only a new last layer (logistic regression on the 512
    embedding numbers) is trained on: the children's training set + n adult X-rays.
    Adults are up-weighted so both groups count equally (capped at 20x).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    ktr = (mk.split == "train").values
    kte = (mk.split == "test").values
    pool = np.flatnonzero(((mr.split == "adapt") & mr.is_adult).values)
    ate = ((mr.split == "test") & mr.is_adult).values
    ate_vs_normal = ate & (mr.cls != "No Lung Opacity / Not Normal").values
    yk, yr = mk.label.values, mr.label.values

    rows = []
    for n in sizes:
        n = min(n, len(pool))
        for seed in range(seeds if 0 < n < len(pool) else 1):
            rng = np.random.default_rng(seed)
            pick = rng.choice(pool, size=n, replace=False) if n else np.array([], dtype=int)
            for strategy in ("children + n adults", "n adults only"):
                if strategy == "n adults only" and (n < 10 or len(np.unique(yr[pick])) < 2):
                    continue
                if strategy == "children + n adults":
                    Xtr = np.concatenate([ek[ktr], er[pick]])
                    ytr = np.concatenate([yk[ktr], yr[pick]])
                    w = np.ones(len(ytr))
                    if n:
                        w[ktr.sum():] = min(20.0, ktr.sum() / n)
                else:
                    Xtr, ytr, w = er[pick], yr[pick], np.ones(n)
                sc = StandardScaler().fit(Xtr)
                clf = LogisticRegression(C=0.1, max_iter=3000).fit(sc.transform(Xtr), ytr, sample_weight=w)
                score = lambda E: clf.decision_function(sc.transform(E))
                rows.append(dict(n_adults=int(n), seed=seed, strategy=strategy,
                                 auroc_adults=metrics.auroc(yr[ate], score(er[ate])),
                                 auroc_adults_vs_normal=metrics.auroc(yr[ate_vs_normal], score(er[ate_vs_normal])),
                                 auroc_children=metrics.auroc(yk[kte], score(ek[kte]))))
    return rows


def make_figures(name, cfg, mk, pk, mr, pr, res, ood_scores, fig_dir):
    import matplotlib.pyplot as plt

    plotting.use_style()
    thr = cfg["threshold"]
    kte = (mk.split == "test").values
    ate = ((mr.split == "test") & mr.is_adult).values
    H = res["headline"]
    kid, adu = H["children_test (Kermany, pneumonia vs normal)"], H["adults_test (RSNA, lung opacity vs everything else)"]

    # 1. headline: same model, same threshold
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    labels = ["Ranking quality (AUROC)", "Pneumonia cases caught\n(sensitivity)", "Healthy correctly cleared\n(specificity)"]
    keys = ["auroc", "sensitivity", "specificity"]
    y = np.arange(len(keys))[::-1]
    for off, grp, col, lab in ((0.19, kid, plotting.CHILDREN, f"Children 1-5, same hospital (n={kid['n']})"),
                               (-0.19, adu, plotting.ADULTS, f"Adults, other hospital (n={adu['n']})")):
        vals = [grp[k] for k in keys]
        ax.barh(y + off, vals, height=0.34, color=col, label=lab)
        for yi, v in zip(y + off, vals):
            ax.text(v + 0.012, yi, f"{v:.2f}", va="center", fontsize=11)
    ax.set_yticks(y, labels); ax.set_xlim(0, 1.12); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.12), ncol=1)
    ax.set_title(res["titles"]["headline"])
    plotting.finish(fig, fig_dir / f"shift_{name}_headline.png",
                    note=f"Model '{name}'. Threshold {thr:.2f} fixed on validation data. Held-out test images only.")

    # 2. what the model says: distribution of predicted probabilities
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5), sharey=False)
    bins = np.linspace(0, 1, 26)
    for ax, y_, p_, col, title in ((axes[0], mk.label.values[kte], pk[kte], plotting.CHILDREN, "Children 1-5 (same hospital)"),
                                   (axes[1], mr.label.values[ate], pr[ate], plotting.ADULTS, "Adults (other hospital)")):
        ax.hist(p_[y_ == 0], bins=bins, color=col, alpha=0.35, label="no pneumonia", density=True)
        ax.hist(p_[y_ == 1], bins=bins, histtype="step", color=col, linewidth=2, label="pneumonia", density=True)
        ax.axvline(thr, color=plotting.NEUTRAL, linestyle="--", linewidth=1.5)
        ax.text(thr, ax.get_ylim()[1] * 0.97, " flagged ->" if thr < 0.6 else "flagged -> ", color=plotting.TEXT_2,
                ha="left" if thr < 0.6 else "right", va="top", fontsize=10)
        ax.set_title(title, fontsize=12); ax.set_xlabel("Model's pneumonia probability"); ax.set_yticks([])
        ax.legend(loc="upper center")
    fig.suptitle(res["titles"]["scores"], x=0.01, ha="left", fontweight="bold", fontsize=14, y=1.03)
    plotting.finish(fig, fig_dir / f"shift_{name}_scores.png", note="Filled = truly no pneumonia, outline = truly pneumonia. Dashed line = decision threshold.")

    # 3. subgroups
    sg = res["subgroups_adult_hospital"]
    names = list(sg)[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 0.42 * len(names) + 1.6))
    for i, nme in enumerate(names):
        s = sg[nme]
        lo, hi = s.get("auroc_ci", (s["auroc"], s["auroc"]))
        ax.plot([lo, hi], [i, i], color=plotting.ADULTS, linewidth=2, alpha=0.5)
        ax.plot(s["auroc"], i, "o", color=plotting.ADULTS, markersize=8)
        ax.text(1.005, i, f"{s['auroc']:.2f}  (n={s['n']})", va="center", fontsize=10, color=plotting.TEXT_2)
    ax.axvline(0.5, color=plotting.NEUTRAL, linestyle="--", linewidth=1.5)
    ax.axvline(kid["auroc"], color=plotting.CHILDREN, linewidth=2)
    ax.text(0.5, len(names) - 0.35, "coin flip", ha="center", fontsize=10, color=plotting.TEXT_2)
    ax.text(kid["auroc"], len(names) - 0.35, "children's test set", ha="right", fontsize=10, color=plotting.TEXT_2)
    ax.set_yticks(range(len(names)), names); ax.set_xlim(0.3, 1.0); ax.set_ylim(-0.6, len(names) + 0.1)
    ax.set_xlabel("AUROC (1.0 = perfect ranking, 0.5 = coin flip)"); ax.grid(axis="y", visible=False)
    ax.set_title(res["titles"]["subgroups"])
    plotting.finish(fig, fig_dir / f"shift_{name}_subgroups.png",
                    note="Other-hospital test set (RSNA), lung opacity vs everything else. Lines = 95% confidence intervals.")

    # 4. the OOD warning
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    sk, sa = ood_scores["children_test"], ood_scores["adults_test"]
    hi = np.percentile(np.concatenate([sk, sa]), 99.5)
    bins = np.linspace(min(sk.min(), sa.min()), hi, 50)
    ax.hist(sk, bins=bins, color=plotting.CHILDREN, alpha=0.6, label="Children 1-5 (same hospital)", density=True)
    ax.hist(sa, bins=bins, color=plotting.ADULTS, alpha=0.6, label="Adults (other hospital)", density=True)
    t95 = res["ood"]["threshold_p95"]
    ax.axvline(t95, color=plotting.NEUTRAL, linestyle="--", linewidth=1.5)
    ax.text(t95, ax.get_ylim()[1] * 0.97, " warning ->", va="top", fontsize=10, color=plotting.TEXT_2)
    ax.set_xlabel("How unfamiliar the X-ray looks to the model (distance to training data)"); ax.set_yticks([])
    ax.legend(loc="upper right"); ax.set_title(res["titles"]["ood"])
    plotting.finish(fig, fig_dir / f"shift_{name}_ood.png",
                    note="Warning threshold = 95th percentile of validation images from the training hospital.")

    # 5. learning curve
    import pandas as pd
    lc = pd.DataFrame(res["learning_curve"])
    lc = lc[lc.strategy == "children + n adults"]
    agg = lc.groupby("n_adults").agg(a=("auroc_adults", "mean"), a_lo=("auroc_adults", "min"), a_hi=("auroc_adults", "max"),
                                     c=("auroc_children", "mean")).reset_index()
    x = np.where(agg.n_adults == 0, 3, agg.n_adults)        # draw n=0 at the left edge of the log axis
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    ax.fill_between(x, agg.a_lo, agg.a_hi, color=plotting.ADULTS, alpha=0.2, linewidth=0)
    ax.plot(x, agg.a, "-o", color=plotting.ADULTS, label="tested on adults (other hospital)")
    ax.plot(x, agg.c, "-o", color=plotting.CHILDREN, label="tested on children 1-5")
    ax.axhline(0.5, color=plotting.NEUTRAL, linestyle="--", linewidth=1.5)
    ax.text(x[-1], 0.505, "coin flip", ha="right", va="bottom", fontsize=10, color=plotting.TEXT_2)
    ax.set_xscale("log"); ax.set_xticks(x, ["0"] + [f"{n:,}" for n in agg.n_adults[1:]], rotation=0, fontsize=10)
    ax.minorticks_off()
    ax.set_xlabel("Labelled adult X-rays added to training"); ax.set_ylabel("AUROC"); ax.set_ylim(0.45, 1.01)
    ax.legend(loc="lower right"); ax.set_title(res["titles"]["learning_curve"])
    plotting.finish(fig, fig_dir / f"shift_{name}_learning_curve.png",
                    note="Network frozen, only the last layer retrained. Shaded band = range over 5 random draws of adult X-rays.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="kids_only")
    ap.add_argument("--no-learning-curve", action="store_true")
    args = ap.parse_args()

    model_dir = C.ARTIFACTS_DIR / args.model
    cfg = json.loads((model_dir / "config.json").read_text())
    T, thr = cfg["temperature"], cfg["threshold"]
    Xk, mk = data.load_kermany()
    Xr, mr = data.load_rsna()
    zk, zr = np.load(model_dir / "preds_kermany.npz"), np.load(model_dir / "preds_rsna.npz")
    pk, pr = modeling.sigmoid(zk["logits"], T), modeling.sigmoid(zr["logits"], T)
    ek, er = zk["emb"], zr["emb"]

    kte = (mk.split == "test").values
    ate = ((mr.split == "test") & mr.is_adult).values
    ava = ((mr.split == "val") & mr.is_adult).values
    res = dict(model=args.model, train_data=cfg.get("train_data"), threshold=thr, temperature=T)

    # 1. headline
    res["headline"] = {k: short(v) for k, v in headline(model_dir, mk, mr).items()}

    # 2. what the fixed threshold does to adults, by RSNA class
    res["adults_flagged_rate_by_class"] = {
        c: dict(n=int((ate & (mr.cls == c).values).sum()), flagged=float((pr[ate & (mr.cls == c).values] >= thr).mean()))
        for c in ("Normal", "No Lung Opacity / Not Normal", "Lung Opacity")}

    # 3. is it only the threshold? Re-choose it on ADULT validation data (uses adult labels, so this is
    #    not something the children-only model could do on its own; it isolates ranking from calibration)
    thr_adult = metrics.threshold_for_sensitivity(mr.label[ava], pr[ava], 0.95)
    res["adults_with_threshold_rechosen_on_adult_val"] = short(metrics.summarize(mr.label[ate], pr[ate], thr_adult, n_boot=500))
    res["adults_with_threshold_rechosen_on_adult_val"]["threshold"] = thr_adult

    # 4. subgroups
    res["subgroups_adult_hospital"] = subgroup_table(mr, pr, thr)

    # 5. OOD warning: fitted on this model's own training images
    if "children" in " ".join(cfg.get("train_data", ["children"])):
        fit_e = [ek[(mk.split == "train").values]]; fit_y = [mk.label.values[(mk.split == "train").values]]
        cal_e = [ek[(mk.split == "val").values]]
    else:
        fit_e, fit_y, cal_e = [], [], []
    if (model_dir / "rsna_train_idx.npy").exists():
        ridx = np.load(model_dir / "rsna_train_idx.npy")
        fit_e.append(er[ridx]); fit_y.append(mr.label.values[ridx]); cal_e.append(er[ava])
    ood = MahalanobisOOD().fit(np.concatenate(fit_e), np.concatenate(fit_y)).calibrate(np.concatenate(cal_e))
    ood.save(model_dir / "ood.npz")
    sk, sa = ood.score(ek[kte]), ood.score(er[ate])
    young = ((mr.split == "test") & mr.age_valid & (mr.age <= 5)).values
    teen = ((mr.split == "test") & mr.age_valid & (mr.age <= 17)).values
    t95, t99 = ood.thresholds["p95"], ood.thresholds["p99"]
    res["ood"] = dict(
        threshold_p95=t95, threshold_p99=t99,
        warned_children_test=float((sk > t95).mean()), warned_adults_test=float((sa > t95).mean()),
        warned_children_test_p99=float((sk > t99).mean()), warned_adults_test_p99=float((sa > t99).mean()),
        warned_other_hospital_age_0_5=dict(n=int(young.sum()), share=float((ood.score(er[young]) > t95).mean())) if young.any() else None,
        warned_other_hospital_age_0_17=dict(n=int(teen.sum()), share=float((ood.score(er[teen]) > t95).mean())) if teen.any() else None,
        auroc_children_vs_adults=metrics.auroc(np.r_[np.zeros(len(sk)), np.ones(len(sa))], np.r_[sk, sa]))
    # does the warning also pick out the adults the model gets wrong?
    wrong = (pr[ate] >= thr).astype(int) != mr.label.values[ate]
    res["ood"]["auroc_score_predicts_error_within_adults"] = metrics.auroc(wrong.astype(int), sa)

    # 6. learning curve
    if not args.no_learning_curve:
        res["learning_curve"] = learning_curve(ek, mk, er, mr, sizes=[0, 10, 30, 100, 300, 1000, 3000, 10000, 10 ** 9])

    kid = res["headline"]["children_test (Kermany, pneumonia vs normal)"]
    adu = res["headline"]["adults_test (RSNA, lung opacity vs everything else)"]
    res["titles"] = dict(
        headline=f"Children: AUROC {kid['auroc']:.2f}. Adults: {adu['auroc']:.2f}. Same model, same threshold.",
        scores=f"The model flags {adu['flagged_rate']:.0%} of adults (children: {kid['flagged_rate']:.0%})",
        subgroups="How well the model ranks X-rays from the other hospital, by group",
        ood=f"The warning fires for {res['ood']['warned_adults_test']:.0%} of adults and {res['ood']['warned_children_test']:.0%} of children",
        learning_curve="How many adult X-rays does it take to fix the model?")

    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if "learning_curve" in res:
        make_figures(args.model, cfg, mk, pk, mr, pr, res, dict(children_test=sk, adults_test=sa), C.FIGURES_DIR)
    (C.RESULTS_DIR / f"shift_{args.model}.json").write_text(json.dumps(res, indent=2))

    # keep the headline numbers inside the model config, so the demo app can show them
    cfg["metrics"] = res["headline"]
    cfg["ood_summary"] = {k: v for k, v in res["ood"].items() if not isinstance(v, dict)}
    (model_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    print(json.dumps({k: res[k] for k in ("headline", "adults_flagged_rate_by_class", "ood")}, indent=1))
    for nme, s in res["subgroups_adult_hospital"].items():
        print(f"{nme:30s} n={s['n']:5d} pos={s['n_pos']:4d} AUROC {s['auroc']:.3f} sens {s['sensitivity']:.3f} spec {s['specificity']:.3f}")
    if "learning_curve" in res:
        import pandas as pd
        print(pd.DataFrame(res["learning_curve"]).groupby(["strategy", "n_adults"])[["auroc_adults", "auroc_adults_vs_normal", "auroc_children"]].mean().round(3).to_string())
