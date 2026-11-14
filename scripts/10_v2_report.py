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
import re
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
from pneumoshift.train import fit_temperature  # noqa: E402

V2 = C.RESULTS_DIR / "v2"
TAG = "v1-before-retraining"
V1 = {"kids_only": "RL8596/pneumoshift-kids-only", "kids_plus_adults": "RL8596/pneumoshift-kids-plus-adults"}
HEADLINE, V1_REPAIR = "v2_kids_plus_all", "kids_plus_adults"
SCRATCH = Path(os.environ.get("PNEUMOSHIFT_SCRATCH", Path(tempfile.gettempdir()) / "pneumoshift_scratch"))
N_BOOT = 1000
# the one recipe of docs/V2_PROTOCOL.md; a run that deviates is not evaluated
RECIPE = dict(epochs=6, lr=3e-4, batch_size=32, mixed_precision=False, epoch_rule="last")
NAME = re.compile(r"^v2_(?:kids_plus_(?P<n>all|\d+)|(?P<adults>adults_all))(?P<r50>_r50)?(?:_s(?P<seed>\d{4}))?$")
N_CHILDREN_TRAIN, N_ADULT_TRAIN = 4578, 15142
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


def exploratory_rows() -> dict:
    """Rows that were added after the first v2 validation result (scripts/11_xray_pretrained_check.py). Always labelled."""
    out = {}
    f = C.RESULTS_DIR / "exploratory" / "xray_pretrained_chex.npz"
    if f.exists():
        z = np.load(f)
        out["xray_pretrained_as_is"] = dict(kind="exploratory", rsna=z["as_is_lung_opacity"].astype(np.float64),
                                            label="EXPLORATORY: public network trained on chest X-rays of another hospital (CheXpert), used as published")
        out["xray_pretrained_frozen"] = dict(kind="exploratory", rsna=z["frozen_features_logreg"].astype(np.float64),
                                             label="EXPLORATORY: the same network + logistic regression trained on the adult training X-rays")
    return out


def parse_name(name: str):
    """v2 run name -> (family without the seed, number of adult training X-rays, seed), or None if it is not a protocol name."""
    m = NAME.match(name)
    if not m:
        return None
    n = N_ADULT_TRAIN if (m["adults"] or m["n"] == "all") else int(m["n"])
    family = re.sub(r"_s\d{4}$", "", name)
    return family, n, int(m["seed"]) if m["seed"] else C.SEED


def check_recipe(name: str, m: dict, mr) -> list:
    """Did this run follow the one recipe of the protocol, and do its rows line up with our tables? -> list of problems."""
    cfg, bad = m["cfg"], []
    parsed = parse_name(name)
    if parsed is None:
        return ["the name is not one of the protocol"]
    _, n_adults, seed = parsed
    for k, want in RECIPE.items():
        if cfg.get(k) != want:
            bad.append(f"{k} is {cfg.get(k)!r}, the protocol says {want!r}")
    if cfg.get("best_epoch") != cfg.get("epochs"):
        bad.append("the kept epoch is not the last one")
    want_arch = "resnet50.tv_in1k" if name.endswith("_r50") or "_r50_" in name else "resnet18.tv_in1k"
    if cfg.get("arch") != want_arch:
        bad.append(f"network is {cfg.get('arch')}, expected {want_arch}")
    if cfg.get("seed") != seed:
        bad.append(f"seed is {cfg.get('seed')}, the name says {seed}")
    if str(cfg.get("device", "cpu")).lower() == "cpu":
        bad.append("trained on a CPU")
    adult_train = ((mr.split == "adapt") & mr.is_adult).values
    idx_file = V2 / name / "rsna_train_idx.npy"
    idx = np.load(idx_file) if idx_file.exists() else np.array([], dtype=int)
    if len(idx) != n_adults:
        bad.append(f"trained on {len(idx)} adult X-rays, the name says {n_adults}")
    if len(idx) and (not adult_train[idx].all() or len(np.unique(idx)) != len(idx)):
        bad.append("its adult training list holds an X-ray that is not an adult training X-ray, or one twice")
    n_children = 0 if name.startswith("v2_adults_all") else N_CHILDREN_TRAIN
    if cfg.get("n_train") != n_children + n_adults:
        bad.append(f"n_train is {cfg.get('n_train')}, expected {n_children + n_adults}")
    val = ((mr.split == "val") & mr.is_adult).values
    stored = (cfg.get("validation") or {}).get("adults", {}).get("auroc")
    got = metrics.auroc(mr.label.values[val], m["rsna"][val])
    if stored is None or abs(got - stored) > 1e-4:          # proves that the PC's rows are our rows
        bad.append(f"adult validation AUROC from its scores is {got:.5f}, its own record says {stored}")
    return bad


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
            # calibration with the model's own temperature (fitted on ITS validation set, see val_data), and with a
            # temperature fitted on adult validation X-rays for every model, so that the rows can be compared
            t_adult = fit_temperature(s[val], yr[val])
            sig_a = lambda z: 1.0 / (1.0 + np.exp(-z / t_adult))  # noqa: E731
            res["calibration_error"] = dict(adults=metrics.ece(yr[test], sig(s[test])), children=metrics.ece(yk[ktest], sig(m["kermany"][ktest])),
                                            adults_with_temperature_from_adult_validation=metrics.ece(yr[test], sig_a(s[test])),
                                            own_temperature_fitted_on=cfg.get("val_data"))
            res["auroc"]["children"] = dict(value=metrics.auroc(yk[ktest], m["kermany"][ktest]))
            res["children_at_stored_threshold"] = rates(yk[ktest], m["kermany"][ktest], point_thr[name]["stored"])
            res["children"] = {rule: rates(yk[ktest], m["kermany"][ktest], thr) for rule, thr in point_thr[name].items()}
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
                 (V1_REPAIR, "kids_only"), (V1_REPAIR, "frozen_imagenet"), ("v2_kids_plus_6000", V1_REPAIR),
                 ("xray_pretrained_frozen", V1_REPAIR), ("xray_pretrained_frozen", "frozen_imagenet"), ("xray_pretrained_as_is", "kids_only")):
        if a in names and b in names:
            out["paired_auroc_differences"][f"{a} minus {b}"] = paired(a, b)
    if HEADLINE in names and V1_REPAIR in names:
        d = out["paired_auroc_differences"][f"{HEADLINE} minus {V1_REPAIR}"]
        p = d["adults"]
        better = bool(p["ci"][0] > 0 and d["adults_PA"]["difference"] > 0 and d["adults_AP"]["difference"] > 0)
        out["primary_comparison"] = dict(
            what=f"adult test AUROC (lung opacity vs everything else), {HEADLINE} minus published {V1_REPAIR}, patients resampled {n_boot} times",
            difference=p["difference"], ci=p["ci"], inside_PA=d["adults_PA"], inside_AP=d["adults_AP"],
            rule="called better only if the overall interval lies above zero and the difference is positive inside both views",
            repair_is_better=better)
    return out


def seeds_and_curve(report: dict) -> dict:
    """Group v2 runs of one recipe over their seeds: mean and range of the runs (not a confidence interval)."""
    fam = {}
    for name, m in report["models"].items():
        parsed = parse_name(name) if m["kind"] == "v2" else None
        if parsed:
            fam.setdefault(parsed[0], dict(n_adults=parsed[1], runs=[]))["runs"].append(name)
    out = {}
    for base, f in sorted(fam.items(), key=lambda kv: (kv[0].startswith("v2_adults"), kv[0].endswith("_r50"), kv[1]["n_adults"])):
        row = dict(runs=sorted(f["runs"]), n_adult_training_xrays=f["n_adults"])
        for g in ("adults", "adults_PA", "adults_AP", "children"):
            vals = [report["models"][n]["auroc"][g]["value"] for n in f["runs"]]
            row[f"auroc_{g}"] = dict(mean=float(np.mean(vals)), min=float(np.min(vals)), max=float(np.max(vals)), n_runs=len(vals))
        out[base] = row
    return out


def fmt(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def markdown(report: dict) -> str:
    M, G = report["models"], report["groups"]
    order = [n for n in ("always_no", "flag_bedside", "metadata_only", "frozen_imagenet", "kids_only", "kids_plus_adults",
                         "v2_kids_plus_6000", HEADLINE, "v2_adults_all", "xray_pretrained_as_is", "xray_pretrained_frozen") if n in M]
    name = lambda n: n + (" (exploratory)" if M[n]["kind"] == "exploratory" else "")  # noqa: E731
    pair = lambda r, g: f"{fmt(r[g]['sensitivity'], 2)} / {fmt(r[g]['specificity'], 2)}"  # noqa: E731
    rng = lambda c: f"{fmt(c[0], 2)} to {fmt(c[1], 2)}"  # noqa: E731
    prev = G["adults"]["prevalence"]
    L = ["# Model comparison (written by scripts/10_v2_report.py)", "",
         "Rules: docs/V2_PROTOCOL.md. Every threshold comes from validation X-rays and is applied unchanged to test X-rays.",
         "Intervals are 95% intervals from resampling patients. 'Lung opacity' is a radiologist's finding, not a confirmed pneumonia.",
         "Operating points: A = 95% sensitivity on adult validation X-rays; B = 95% sensitivity on children's + adult validation X-rays",
         "pooled (shown without interval); C = 95% specificity on adult validation X-rays (added 14 Nov, after the v1 test results).",
         "Rows marked exploratory were added after the first v2 validation result and are not part of the protocol.", "",
         f"## Adults, lung opacity vs everything else (n = {G['adults']['n']:,} X-rays of {G['adults']['n_patients']:,} patients, "
         f"{100 * prev:.1f}% with lung opacity)", "",
         "| Model | AUROC | inside standing (PA) | inside bedside (AP) | A: sens / spec | B: sens / spec | C: sens / spec |",
         "|---|---|---|---|---|---|---|"]
    for n in order:
        m, a = M[n], M[n]["auroc"]
        L.append(f"| {name(n)} | {fmt(a['adults']['value'])} ({fmt(a['adults']['ci'][0])} to {fmt(a['adults']['ci'][1])}) | "
                 f"{fmt(a['adults_PA']['value'])} | {fmt(a['adults_AP']['value'])} | {pair(m['rules']['A'], 'adults')} | "
                 f"{pair(m['rules']['B'], 'adults') if 'B' in m['rules'] else 'n/a'} | {pair(m['rules']['C'], 'adults')} |")
    L += ["", f"## Standing X-rays only (PA): n = {G['adults_PA']['n']:,}, {100 * G['adults_PA']['prevalence']:.1f}% with lung opacity "
          f"('always no' is right for {100 * (1 - G['adults_PA']['prevalence']):.1f}%)", "",
          "| Model | AUROC | A: sensitivity | A: specificity | C: sensitivity | C: specificity |", "|---|---|---|---|---|---|"]
    for n in order:
        m, a = M[n], M[n]["auroc"]["adults_PA"]
        cell = lambda r, k: f"{fmt(m['rules'][r]['adults_PA'][k], 2)} ({rng(m['rules'][r]['adults_PA'][k + '_ci'])})"  # noqa: E731
        L.append(f"| {name(n)} | {fmt(a['value'])} ({fmt(a['ci'][0])} to {fmt(a['ci'][1])}) | {cell('A', 'sensitivity')} | "
                 f"{cell('A', 'specificity')} | {cell('C', 'sensitivity')} | {cell('C', 'specificity')} |")
    L += ["", "## Adults at point A, with intervals, and further checks", "",
          f"'Always no' is right for {100 * (1 - prev):.1f}% of these adults. Curve read-offs describe the test curve; they are not thresholds set in advance.", "",
          "| Model | A: sensitivity | A: specificity | A: right answers | Read off the curve: spec. at 95% sens. | sens. at 95% spec. | "
          "Bedside vs standing among Normal adults (0.5 = the score ignores the view) |", "|---|---|---|---|---|---|---|"]
    for n in order:
        m = M[n]
        r = m["rules"]["A"]["adults"]
        cr = m["curve_reads"].get("adults", {})
        L.append(f"| {name(n)} | {fmt(r['sensitivity'], 2)} ({rng(r['sensitivity_ci'])}) | {fmt(r['specificity'], 2)} ({rng(r['specificity_ci'])}) | "
                 f"{fmt(r['accuracy'], 3)} | {fmt(cr.get('specificity_at_95_sensitivity'), 2)} | {fmt(cr.get('sensitivity_at_95_specificity'), 2)} | "
                 f"{fmt(m['view_shortcut_auroc_among_normal_adults'], 2)} |")
    nets = [n for n in order if "calibration_error" in M[n]]
    L += ["", "## Children, and calibration", "",
          "'Own threshold' is the one stored with the model; the validation X-rays it was chosen on differ between models (last column).", "",
          "| Model | Children: AUROC | at own threshold: sens / spec | at point A: sens / spec | Calibration error, adults (own temperature) | "
          "adults (temperature from adult validation) | children | Own threshold and temperature chosen on |", "|---|---|---|---|---|---|---|---|"]
    for n in nets:
        m, c = M[n], M[n]["calibration_error"]
        L.append(f"| {n} | {fmt(m['auroc']['children']['value'])} | {pair(m['children'], 'stored')} | {pair(m['children'], 'A')} | {fmt(c['adults'])} | "
                 f"{fmt(c['adults_with_temperature_from_adult_validation'])} | {fmt(c['children'])} | {c['own_temperature_fitted_on']} |")
    if "primary_comparison" in report:
        p = report["primary_comparison"]
        L += ["", "## Primary comparison (the only confirmation in the protocol)", "",
              f"{p['what']}: **{p['difference']:+.3f}** (95% interval {p['ci'][0]:+.3f} to {p['ci'][1]:+.3f}). "
              f"Inside standing X-rays {p['inside_PA']['difference']:+.3f} ({p['inside_PA']['ci'][0]:+.3f} to {p['inside_PA']['ci'][1]:+.3f}), "
              f"inside bedside X-rays {p['inside_AP']['difference']:+.3f} ({p['inside_AP']['ci'][0]:+.3f} to {p['inside_AP']['ci'][1]:+.3f}). "
              f"Rule: {p['rule']}. **Verdict: " + ("the repaired model is better.**" if p["repair_is_better"] else "no improvement is shown.**")]
    if report.get("paired_auroc_differences"):
        L += ["", "## Other paired differences in adult AUROC (description, not confirmation)", "", "| Comparison | Adults | inside PA | inside AP |", "|---|---|---|---|"]
        for k, d in report["paired_auroc_differences"].items():
            c = lambda g: f"{d[g]['difference']:+.3f} ({d[g]['ci'][0]:+.3f} to {d[g]['ci'][1]:+.3f})"  # noqa: E731
            L.append(f"| {k} | {c('adults')} | {c('adults_PA')} | {c('adults_AP')} |")
    if report.get("seeds"):
        L += ["", "## Repair curve and seeds (mean and range of the runs, not a confidence interval)", "",
              "At 300 to 6,000 adult X-rays every seed draws its own X-rays, so the range covers the draw and the training.", "",
              "| Recipe | adult training X-rays | runs | adult AUROC | standing (PA) | bedside (AP) | children |", "|---|---|---|---|---|---|---|"]
        for base, r in report["seeds"].items():
            c = lambda g: f"{fmt(r[f'auroc_{g}']['mean'])} ({fmt(r[f'auroc_{g}']['min'])} to {fmt(r[f'auroc_{g}']['max'])})"  # noqa: E731
            L.append(f"| {base} | {r['n_adult_training_xrays']:,} | {len(r['runs'])} | {c('adults')} | {c('adults_PA')} | {c('adults_AP')} | {c('children')} |")
        for n, label in (("kids_only", "published v1, best epoch kept"), ("kids_plus_adults", "published v1, 6,000 adults, best epoch kept")):
            if n in M:
                a = M[n]["auroc"]
                L.append(f"| {n} ({label}) | {'0' if n == 'kids_only' else '6,000'} | 1 | {fmt(a['adults']['value'])} | {fmt(a['adults_PA']['value'])} | "
                         f"{fmt(a['adults_AP']['value'])} | {fmt(a['children']['value'])} |")
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
    ap.add_argument("--again", default=None, metavar="REASON",
                    help="compute the v2 test results a second time; the reason is written into the report")
    args = ap.parse_args()

    Xk, mk = data.load_kermany()
    Xr, mr = data.load_rsna()

    if args.what == "val":
        v2 = load_v2()
        if not v2:
            sys.exit("no finished v2 run in results/v2/")
        problems = {name: check_recipe(name, m, mr) for name, m in v2.items()}
        problems = {k: v for k, v in problems.items() if v}
        if problems:
            for k, v in problems.items():
                print(f"NOT ACCORDING TO THE PROTOCOL: {k}: " + "; ".join(v))
            sys.exit("\nThese runs are not evaluated. Move their folders out of results/v2/ (and say so in the write-up), then run this again.")
        rows = validation_summary(v2, mk, mr)
        out = dict(recipe_check=f"all {len(v2)} runs follow the recipe of docs/V2_PROTOCOL.md; their rows line up with our tables",
                   note="Validation X-rays only. Written before any test result of a v2 model is computed. "
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
        v2 = load_v2()
        listed = set(json.loads((C.RESULTS_DIR / "v2_validation.json").read_text())["runs"])
        if listed != set(v2):
            sys.exit(f"results/v2_validation.json lists other runs than results/v2/ holds (only in one of them: {sorted(listed ^ set(v2))}). "
                     "Run 'val' again and commit before 'test'.")
        if (C.RESULTS_DIR / "v2_report.json").exists() and not args.again:
            sys.exit("results/v2_report.json exists: the v2 test results were already computed. The protocol says once. "
                     "If there is a reason (for example a corrected script), pass --again \"the reason\".")
        models.update(v2)
    adult = mr.is_adult.values
    train = ((mr.split == "adapt") & adult).values
    val = ((mr.split == "val") & adult).values
    log("baselines trained on the adult training X-rays")
    models.update(baselines(mr, Xr, train, val))
    models.update(exploratory_rows())
    log(f"evaluating {len(models)} models and baselines, {args.boot} patient-level resamples")
    report = evaluate(models, mk, mr, args.boot)
    report["seeds"] = seeds_and_curve(report)
    report["evaluated_models"] = sorted(models)
    report["computed_at_utc"] = time.strftime("%Y-%m-%d %H:%M", time.gmtime())
    if args.again:
        report["computed_again_because"] = args.again
    report["note"] = ("Test results. Thresholds come from validation X-rays only (docs/V2_PROTOCOL.md). 'Lung opacity' is a radiologist's "
                      "finding, not a confirmed pneumonia. Seeds: range of the runs, not a confidence interval.")
    stem = args.out or ("v1_rehearsal" if args.v1_only else "v2_report")
    (C.RESULTS_DIR / f"{stem}.json").write_text(json.dumps(report, indent=1))
    (C.RESULTS_DIR / f"{stem}.md").write_text(markdown(report))
    log(f"wrote results/{stem}.json and results/{stem}.md")
    print(markdown(report))
