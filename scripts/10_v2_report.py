"""Step 10: the final comparison of all models, by the rules fixed in docs/V2_PROTOCOL.md.

    python scripts/10_v2_report.py val               # validation summary of the finished v2 runs (no test data)
    python scripts/10_v2_report.py test              # test results of every model, computed here once
    python scripts/10_v2_report.py test --v1-only    # rehearsal with the two published models and the baselines

Runs in the cloud session, never on the training machine. It needs no GPU and no network weights: it works
from each model's stored score for every X-ray (results/v2/<name>/logits.npz for v2 models, the Hub files of
the published models at the tag v1-before-retraining) and from the prepared data tables.

What it computes, for every model and for the baselines trained on the same adult X-rays:

  - AUROC on the adult test X-rays (lung opacity vs everything else), overall and inside each X-ray view,
    with 95% intervals from resampling PATIENTS (one patient can have several X-rays)
  - the three operating points of the protocol (thresholds from validation X-rays only):
        A  95% sensitivity on adult validation X-rays (screening)
        B  95% sensitivity on children's + adult validation X-rays pooled (as first implemented)
        C  95% specificity on adult validation X-rays (high confidence; added 14 Nov, disclosed)
    Intervals resample validation patients and test patients together, so they include the threshold's own uncertainty.
  - two summaries read off the test curve (not thresholds anyone could set in advance):
    specificity at 95% sensitivity, sensitivity at 95% specificity
  - the shortcut check: how well the score tells bedside from standing X-rays among Normal adults
  - calibration error, and results on children at the model's own stored threshold
  - the primary comparison of the protocol: v2_kids_plus_all against the published kids_plus_adults,
    paired difference of adult AUROC with a patient-level interval

"test" refuses to include v2 models until results/v2_validation.json is committed, so the order
"validation first, test once" is visible in the git history.

Outputs: results/v2_validation.json (val), results/v2_report.json and results/v2_report.md (test).
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pneumoshift import config as C, data, metrics  # noqa: E402

V2 = C.RESULTS_DIR / "v2"
TAG = "v1-before-retraining"
V1 = {"kids_only": "RL8596/pneumoshift-kids-only", "kids_plus_adults": "RL8596/pneumoshift-kids-plus-adults"}
HEADLINE, V1_REPAIR = "v2_kids_plus_all", "kids_plus_adults"
SCRATCH = Path(os.environ.get("PNEUMOSHIFT_SCRATCH", Path(tempfile.gettempdir()) / "pneumoshift_scratch"))
N_BOOT = 1000
SEED = 2026
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:5.0f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------------
# Small statistics helpers. Scores are raw model outputs (logits): every rule below only uses their order.
# ---------------------------------------------------------------------------------------------------
def auroc_rows(S: np.ndarray, y: np.ndarray) -> np.ndarray:
    """AUROC of every row of S [n_models, n] against labels y [n]. NaN if y holds one class only."""
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.full(S.shape[0], np.nan)
    r = rankdata(S, axis=1)
    return (r[:, y == 1].sum(axis=1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def thr_sensitivity(y, s, target=0.95) -> float:
    """Highest threshold that still flags at least `target` of the cases (flag = score >= threshold)."""
    pos = np.sort(s[y == 1])
    return float(pos[int(np.floor((1 - target) * len(pos)))])


def thr_specificity(y, s, target=0.95) -> float:
    """Lowest threshold at which at least `target` of the non-cases are not flagged."""
    neg = np.sort(s[y == 0])
    k = int(np.ceil(target * len(neg)))
    return float(np.nextafter(neg[k - 1], np.inf))


def rates(y, s, thr) -> dict:
    flag = s >= thr
    tp = int((flag & (y == 1)).sum()); fn = int((~flag & (y == 1)).sum())
    tn = int((~flag & (y == 0)).sum()); fp = int((flag & (y == 0)).sum())
    div = lambda a, b: float(a / b) if b else float("nan")  # noqa: E731
    return dict(sensitivity=div(tp, tp + fn), specificity=div(tn, tn + fp), accuracy=div(tp + tn, len(y)),
                flagged=div(tp + fp, len(y)), ppv=div(tp, tp + fp), npv=div(tn, tn + fn), tp=tp, fp=fp, tn=tn, fn=fn)


def curve_reads(y, s) -> dict:
    """Read off the curve of THIS data: specificity at 95% sensitivity and sensitivity at 95% specificity."""
    return dict(specificity_at_95_sensitivity=rates(y, s, thr_sensitivity(y, s))["specificity"],
                sensitivity_at_95_specificity=rates(y, s, thr_specificity(y, s))["sensitivity"])


def ci(values) -> list:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [float("nan")] * 2


class PatientSampler:
    """Draws bootstrap samples of whole patients from the rows where `mask` is true."""

    def __init__(self, patients: np.ndarray, mask: np.ndarray):
        self.rows = np.flatnonzero(mask)
        ids, inverse = np.unique(patients[self.rows], return_inverse=True)
        order = np.argsort(inverse, kind="stable")
        self.sorted_rows = self.rows[order]
        self.starts = np.searchsorted(inverse[order], np.arange(len(ids) + 1))
        self.n_patients = len(ids)

    def draw(self, rng) -> np.ndarray:
        pick = rng.integers(0, self.n_patients, self.n_patients)
        return np.concatenate([self.sorted_rows[self.starts[p]:self.starts[p + 1]] for p in pick])


# ---------------------------------------------------------------------------------------------------
# Models and baselines: a score for every adult-hospital row (and for every children's row, where it exists)
# ---------------------------------------------------------------------------------------------------
def load_v1() -> dict:
    from huggingface_hub import hf_hub_download

    out = {}
    for name, repo in V1.items():
        cfg = json.loads(Path(hf_hub_download(repo, "config.json", revision=TAG)).read_text())
        out[name] = dict(kind="published v1", cfg=cfg,
                         rsna=np.load(hf_hub_download(repo, "analysis/preds_rsna.npz", revision=TAG))["logits"].astype(np.float64),
                         kermany=np.load(hf_hub_download(repo, "analysis/preds_kermany.npz", revision=TAG))["logits"].astype(np.float64))
    return out


def load_v2() -> dict:
    out = {}
    for cfg_path in sorted(V2.glob("*/config.json")):
        z = np.load(cfg_path.parent / "logits.npz")
        out[cfg_path.parent.name] = dict(kind="v2", cfg=json.loads(cfg_path.read_text()),
                                         rsna=z["rsna"].astype(np.float64), kermany=z["kermany"].astype(np.float64))
    return out


def stored_threshold_logit(cfg: dict) -> float:
    """The model's own threshold (a probability after temperature scaling), as a raw score."""
    p, t = float(cfg["threshold"]), float(cfg.get("temperature", 1.0))
    return float(t * np.log(p / (1 - p)))


def baselines(mr, Xr, train, val) -> dict:
    """Simple rules and models that are trained on the SAME adult training X-rays as the repaired model."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    y = mr.label.values.astype(int)
    out = {"always_no": dict(kind="baseline", label="Always answers 'no lung opacity'", rsna=np.zeros(len(mr)), fixed_threshold=1.0),
           "flag_bedside": dict(kind="baseline", label="Flags every bedside X-ray, looks at no picture",
                                rsna=(mr["view"] == "AP").values.astype(float), fixed_threshold=0.5)}

    meta = np.column_stack([(mr["view"] == "AP").values, mr.age.clip(0, 100).values / 100.0, (mr.sex == "M").values]).astype(float)
    clf = LogisticRegression(max_iter=2000).fit(meta[train], y[train])
    out["metadata_only"] = dict(kind="baseline", label="Logistic regression on view, age and sex (no picture), trained on the adult training X-rays",
                                rsna=clf.decision_function(meta))

    cache = SCRATCH / "imagenet_resnet18_rsna_emb.npy"
    if cache.exists() and len(np.load(cache, mmap_mode="r")) == len(mr):
        emb = np.load(cache)
    else:
        import torch

        from pneumoshift import modeling

        log(f"computing frozen ImageNet features for {len(mr)} adult-hospital X-rays (a few minutes on a CPU, cached afterwards)")
        torch.manual_seed(0)
        net = modeling.build_model(pretrained=True).eval()
        _, emb = modeling.predict(net, Xr, return_embeddings=True)
        SCRATCH.mkdir(parents=True, exist_ok=True)
        np.save(cache, emb.astype(np.float32))
    sc = StandardScaler().fit(emb[train])
    best = None
    for c in (1e-3, 1e-2, 1e-1, 1.0):                      # the strength is chosen on adult VALIDATION X-rays
        clf = LogisticRegression(C=c, max_iter=3000).fit(sc.transform(emb[train]), y[train])
        a = metrics.auroc(y[val], clf.decision_function(sc.transform(emb[val])))
        if best is None or a > best[0]:
            best = (a, c, clf)
    out["frozen_imagenet"] = dict(kind="baseline", chosen_C=best[1],
                                  label="Frozen ImageNet ResNet-18 + logistic regression, trained on the adult training X-rays",
                                  rsna=best[2].decision_function(sc.transform(emb)))
    return out


# ---------------------------------------------------------------------------------------------------
def validation_summary(models: dict, mk, mr) -> dict:
    yr, yk = mr.label.values.astype(int), mk.label.values.astype(int)
    val = ((mr.split == "val") & mr.is_adult).values
    kval = (mk.split == "val").values
    rows = {}
    for name, m in models.items():
        s = m["rsna"]
        row = dict(adult_val_auroc=metrics.auroc(yr[val], s[val]), children_val_auroc=metrics.auroc(yk[kval], m["kermany"][kval]),
                   kept_epoch=m["cfg"].get("best_epoch"), epochs=m["cfg"].get("epochs"), seed=m["cfg"].get("seed"),
                   n_train=m["cfg"].get("n_train"), train_data=m["cfg"].get("train_data"), arch=m["cfg"].get("arch"))
        for view in ("PA", "AP"):
            v = val & (mr["view"] == view).values
            row[f"adult_val_auroc_{view}"] = metrics.auroc(yr[v], s[v])
        rows[name] = row
    return rows


def evaluate(models: dict, mk, mr, n_boot: int) -> dict:
    yr, yk = mr.label.values.astype(int), mk.label.values.astype(int)
    adult = mr.is_adult.values
    val = (mr.split == "val").values & adult
    test = (mr.split == "test").values & adult
    kval, ktest = (mk.split == "val").values, (mk.split == "test").values
    is_pa, is_ap = (mr["view"] == "PA").values, (mr["view"] == "AP").values
    normal = (mr.cls == "Normal").values
    groups = {"adults": test, "adults_PA": test & is_pa, "adults_AP": test & is_ap,
              "adults_vs_normal": test & (mr.cls != "No Lung Opacity / Not Normal").values}
    patients = mr.nih_patient.astype(str).values
    names = list(models)
    S = np.stack([models[n]["rsna"] for n in names])

    # thresholds from validation data (per model)
    def thresholds(name, val_rows, kval_rows=None):
        m, s = models[name], models[name]["rsna"]
        if "fixed_threshold" in m:
            return dict(A=m["fixed_threshold"], C=m["fixed_threshold"])
        t = dict(A=thr_sensitivity(yr[val_rows], s[val_rows]), C=thr_specificity(yr[val_rows], s[val_rows]))
        if m.get("kermany") is not None and kval_rows is not None:
            pooled_y = np.concatenate([yk[kval_rows], yr[val_rows]])
            pooled_s = np.concatenate([m["kermany"][kval_rows], s[val_rows]])
            t["B"] = thr_sensitivity(pooled_y, pooled_s)
        return t

    val_rows, kval_rows = np.flatnonzero(val), np.flatnonzero(kval)
    out = {"groups": {g: dict(n=int(m.sum()), n_pos=int(yr[m].sum()), n_patients=int(len(np.unique(patients[m]))),
                              prevalence=float(yr[m].mean())) for g, m in groups.items()},
           "validation": dict(adults=dict(n=int(val.sum()), n_pos=int(yr[val].sum()), n_patients=int(len(np.unique(patients[val]))))),
           "models": {}}

    point_thr = {n: thresholds(n, val_rows, kval_rows) for n in names}
    for i, name in enumerate(names):
        m, s = models[name], models[name]["rsna"]
        res = dict(kind=m["kind"], label=m.get("label"), auroc={}, rules={}, curve_reads={})
        if "cfg" in m:
            cfg = m["cfg"]
            res.update(train_data=cfg.get("train_data"), seed=cfg.get("seed"), arch=cfg.get("arch"), n_train=cfg.get("n_train"),
                       kept_epoch=cfg.get("best_epoch"), epoch_rule=cfg.get("epoch_rule", "best"), val_data=cfg.get("val_data"))
            point_thr[name]["stored"] = stored_threshold_logit(cfg)
            temp = float(cfg.get("temperature", 1.0))
            sig = lambda z: 1.0 / (1.0 + np.exp(-z / temp))  # noqa: E731
            res["calibration_error"] = dict(adults=metrics.ece(yr[test], sig(s[test])), children=metrics.ece(yk[ktest], sig(m["kermany"][ktest])))
            res["auroc"]["children"] = dict(value=metrics.auroc(yk[ktest], m["kermany"][ktest]))
            res["children_at_stored_threshold"] = rates(yk[ktest], m["kermany"][ktest], point_thr[name]["stored"])
        for g, mask in groups.items():
            res["auroc"][g] = dict(value=float(auroc_rows(s[None, mask], yr[mask])[0]))
            if "fixed_threshold" not in m:
                res["curve_reads"][g] = curve_reads(yr[mask], s[mask])
        for rule, thr in point_thr[name].items():
            res["rules"][rule] = dict(threshold=thr, **{g: rates(yr[mask], s[mask], thr) for g, mask in groups.items() if g != "adults_vs_normal"})
        nrm = test & normal
        res["view_shortcut_auroc_among_normal_adults"] = float(auroc_rows(s[None, nrm], is_ap[nrm].astype(int))[0])
        out["models"][name] = res

    # patient-level bootstrap: test patients and validation patients are resampled together
    rng = np.random.default_rng(SEED)
    samplers = {g: PatientSampler(patients, mask) for g, mask in groups.items()}
    val_sampler = PatientSampler(patients, val)
    boot_auc = {g: np.full((n_boot, len(names)), np.nan) for g in groups}
    boot_rule = {(n, r, g, k): [] for n in names for r in ("A", "C") for g in ("adults", "adults_PA") for k in ("sensitivity", "specificity")}
    all_sampler = samplers["adults"]
    for b in range(n_boot):
        idx = all_sampler.draw(rng)
        y_b = yr[idx]
        boot_auc["adults"][b] = auroc_rows(S[:, idx], y_b)
        for g, mask in (("adults_PA", is_pa), ("adults_AP", is_ap), ("adults_vs_normal", groups["adults_vs_normal"])):
            sub = idx[mask[idx]]
            boot_auc[g][b] = auroc_rows(S[:, sub], yr[sub])
        vidx = val_sampler.draw(rng)
        pa = idx[is_pa[idx]]
        for name in names:
            t = thresholds(name, vidx)
            s = models[name]["rsna"]
            for rule in ("A", "C"):
                for g, rows in (("adults", idx), ("adults_PA", pa)):
                    r = rates(yr[rows], s[rows], t[rule])
                    boot_rule[(name, rule, g, "sensitivity")].append(r["sensitivity"])
                    boot_rule[(name, rule, g, "specificity")].append(r["specificity"])
    for i, name in enumerate(names):
        for g in groups:
            out["models"][name]["auroc"][g]["ci"] = ci(boot_auc[g][:, i])
        for rule in ("A", "C"):
            for g in ("adults", "adults_PA"):
                for k in ("sensitivity", "specificity"):
                    out["models"][name]["rules"][rule][g][k + "_ci"] = ci(boot_rule[(name, rule, g, k)])

    def paired(a, b):
        ia, ib = names.index(a), names.index(b)
        return {g: dict(difference=out["models"][a]["auroc"][g]["value"] - out["models"][b]["auroc"][g]["value"],
                        ci=ci(boot_auc[g][:, ia] - boot_auc[g][:, ib])) for g in ("adults", "adults_PA", "adults_AP")}

    out["paired_auroc_differences"] = {}
    for a, b in ((HEADLINE, V1_REPAIR), (HEADLINE, "v2_kids_plus_6000"), (HEADLINE, "v2_adults_all"), (HEADLINE, "frozen_imagenet"),
                 (V1_REPAIR, "kids_only"), (V1_REPAIR, "frozen_imagenet")):
        if a in names and b in names:
            out["paired_auroc_differences"][f"{a} minus {b}"] = paired(a, b)
    if HEADLINE in names and V1_REPAIR in names:
        p = out["paired_auroc_differences"][f"{HEADLINE} minus {V1_REPAIR}"]["adults"]
        out["primary_comparison"] = dict(
            what=f"adult test AUROC (lung opacity vs everything else), {HEADLINE} minus published {V1_REPAIR}, patients resampled {n_boot} times",
            difference=p["difference"], ci=p["ci"], interval_excludes_zero=bool(p["ci"][0] > 0 or p["ci"][1] < 0))
    return out


def seeds_and_curve(report: dict) -> dict:
    """Group v2 runs of one recipe over their seeds: mean and range of three runs (not a confidence interval)."""
    fam = {}
    for name, m in report["models"].items():
        if m["kind"] != "v2":
            continue
        base = name.rsplit("_s", 1)[0] if name.rsplit("_s", 1)[-1].isdigit() else name
        fam.setdefault(base, []).append(name)
    out = {}
    for base, members in sorted(fam.items()):
        row = dict(runs=sorted(members))
        for g in ("adults", "adults_PA", "adults_AP", "children"):
            vals = [report["models"][n]["auroc"][g]["value"] for n in members]
            row[f"auroc_{g}"] = dict(mean=float(np.mean(vals)), min=float(np.min(vals)), max=float(np.max(vals)), n_runs=len(vals))
        out[base] = row
    return out


def fmt(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def markdown(report: dict) -> str:
    M = report["models"]
    G = report["groups"]
    order = [n for n in ("always_no", "flag_bedside", "metadata_only", "frozen_imagenet", "kids_only", "kids_plus_adults",
                         "v2_kids_plus_6000", HEADLINE, "v2_adults_all") if n in M]
    L = ["# Model comparison (written by scripts/10_v2_report.py)", "",
         "Rules: docs/V2_PROTOCOL.md. Every threshold comes from validation X-rays. Intervals: 95%, patients resampled.", "",
         f"## Adults, lung opacity vs everything else (n = {G['adults']['n']:,} X-rays of {G['adults']['n_patients']:,} patients, "
         f"{100 * G['adults']['prevalence']:.1f}% with lung opacity)", "",
         "| Model | AUROC | inside standing (PA) | inside bedside (AP) | A: sensitivity / specificity | C: sensitivity / specificity |",
         "|---|---|---|---|---|---|"]
    for n in order:
        m = M[n]
        a = m["auroc"]
        cell = lambda r: f"{fmt(m['rules'][r]['adults']['sensitivity'], 2)} / {fmt(m['rules'][r]['adults']['specificity'], 2)}"  # noqa: E731
        L.append(f"| {n} | {fmt(a['adults']['value'])} ({fmt(a['adults']['ci'][0])} to {fmt(a['adults']['ci'][1])}) | "
                 f"{fmt(a['adults_PA']['value'])} | {fmt(a['adults_AP']['value'])} | {cell('A')} | {cell('C')} |")
    L += ["", f"## Standing X-rays only (PA): n = {G['adults_PA']['n']:,}, {100 * G['adults_PA']['prevalence']:.1f}% with lung opacity", "",
          "| Model | AUROC | A: sensitivity / specificity | C: sensitivity / specificity |", "|---|---|---|---|"]
    for n in order:
        m = M[n]
        a = m["auroc"]["adults_PA"]
        cell = lambda r: f"{fmt(m['rules'][r]['adults_PA']['sensitivity'], 2)} / {fmt(m['rules'][r]['adults_PA']['specificity'], 2)}"  # noqa: E731
        L.append(f"| {n} | {fmt(a['value'])} ({fmt(a['ci'][0])} to {fmt(a['ci'][1])}) | {cell('A')} | {cell('C')} |")
    if "primary_comparison" in report:
        p = report["primary_comparison"]
        L += ["", "## Primary comparison", "", f"{p['what']}: **{p['difference']:+.3f}** (95% interval {p['ci'][0]:+.3f} to {p['ci'][1]:+.3f}). "
              + ("The interval excludes zero." if p["interval_excludes_zero"] else "The interval includes zero: no difference is shown.")]
    if report.get("seeds"):
        L += ["", "## Runs of one recipe (mean and range over seeds, not a confidence interval)", "",
              "| Recipe | runs | adult AUROC | standing (PA) | bedside (AP) | children |", "|---|---|---|---|---|---|"]
        for base, r in report["seeds"].items():
            c = lambda g: f"{fmt(r[f'auroc_{g}']['mean'])} ({fmt(r[f'auroc_{g}']['min'])} to {fmt(r[f'auroc_{g}']['max'])})"  # noqa: E731
            L.append(f"| {base} | {len(r['runs'])} | {c('adults')} | {c('adults_PA')} | {c('adults_AP')} | {c('children')} |")
    return "\n".join(L) + "\n"


def committed(path: Path) -> bool:
    rel = str(path.relative_to(ROOT)).replace("\\", "/")
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT).returncode == 0
    return tracked and clean


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["val", "test"])
    ap.add_argument("--v1-only", action="store_true", help="only the published models and the baselines (a rehearsal)")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--out", default=None, help="file name stem in results/ (default v2_report, or v1_rehearsal with --v1-only)")
    args = ap.parse_args()

    Xk, mk = data.load_kermany()
    Xr, mr = data.load_rsna()

    if args.what == "val":
        v2 = load_v2()
        if not v2:
            sys.exit("no finished v2 run in results/v2/")
        rows = validation_summary(v2, mk, mr)
        out = dict(note="Validation X-rays only. Written before any test result of a v2 model is computed. "
                        "The headline model is fixed by name in docs/V2_PROTOCOL.md; nothing is chosen from this table.",
                   headline=HEADLINE, runs=rows)
        (C.RESULTS_DIR / "v2_validation.json").write_text(json.dumps(out, indent=1))
        for name, r in rows.items():
            print(f"{name:28s} adult validation AUROC {r['adult_val_auroc']:.4f}  PA {r['adult_val_auroc_PA']:.4f}  AP {r['adult_val_auroc_AP']:.4f}  "
                  f"children {r['children_val_auroc']:.4f}  kept epoch {r['kept_epoch']}")
        print(f"\nwrote results/v2_validation.json ({len(rows)} runs). Commit it, then run: python scripts/10_v2_report.py test")
        sys.exit(0)

    models = load_v1()
    if not args.v1_only:
        if not committed(C.RESULTS_DIR / "v2_validation.json"):
            sys.exit("results/v2_validation.json is not committed. Run 'python scripts/10_v2_report.py val', commit the file, then run this again.")
        models.update(load_v2())
    adult = mr.is_adult.values
    train = ((mr.split == "adapt") & adult).values
    val = ((mr.split == "val") & adult).values
    log("baselines trained on the adult training X-rays")
    models.update(baselines(mr, Xr, train, val))
    log(f"evaluating {len(models)} models and baselines, {args.boot} patient-level resamples")
    report = evaluate(models, mk, mr, args.boot)
    report["seeds"] = seeds_and_curve(report)
    report["note"] = ("Test results. Thresholds come from validation X-rays only (docs/V2_PROTOCOL.md). 'Lung opacity' is a radiologist's "
                      "finding, not a confirmed pneumonia. Seeds: range of the runs, not a confidence interval.")
    stem = args.out or ("v1_rehearsal" if args.v1_only else "v2_report")
    (C.RESULTS_DIR / f"{stem}.json").write_text(json.dumps(report, indent=1))
    (C.RESULTS_DIR / f"{stem}.md").write_text(markdown(report))
    log(f"wrote results/{stem}.json and results/{stem}.md")
    print(markdown(report))
