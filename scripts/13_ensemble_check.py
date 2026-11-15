"""Step 13 (exploratory): does averaging the networks we already trained give a better adult score?

    python scripts/13_ensemble_check.py val     # validation X-rays only
    python scripts/13_ensemble_check.py test    # test X-rays, once

Rules: docs/V2_ADDENDUM_ENSEMBLE.md, written and committed before this script was run. No training happens
here: the script only averages the stored scores of finished v2 runs (results/v2/<name>/logits.npz).
It was added on 15 Nov 2026, AFTER the test results of the single models were seen, so its result is
exploratory and never replaces the primary comparison of docs/V2_PROTOCOL.md.

The statistics are the functions of scripts/10_v2_report.py, with the same patient resamples (same random
seed, same order of draws). As a check, this script recomputes the report's primary comparison and stops if
it does not get the same numbers.

Outputs: results/exploratory/ensemble_validation.json (val), results/exploratory/ensemble.json and .md (test).
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pneumoshift import config as C, data, metrics  # noqa: E402
from pneumoshift.train import fit_temperature  # noqa: E402

_spec = importlib.util.spec_from_file_location("v2_report", ROOT / "scripts" / "10_v2_report.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

OUT = C.RESULTS_DIR / "exploratory"
ADDENDUM = ROOT / "docs" / "V2_ADDENDUM_ENSEMBLE.md"
# fixed by name in the addendum; nothing is selected
ENSEMBLES = {
    "ensemble_3": ["v2_kids_plus_all", "v2_kids_plus_all_s2027", "v2_kids_plus_all_s2028"],
    "ensemble_7": ["v2_kids_plus_all", "v2_kids_plus_all_s2027", "v2_kids_plus_all_s2028",
                   "v2_adults_all", "v2_adults_all_s2027", "v2_adults_all_s2028", "v2_kids_plus_all_r50"],
}
MAIN, PUBLISHED, SINGLE = "ensemble_3", "kids_plus_adults", "v2_kids_plus_all"
PAIRS = [(MAIN, PUBLISHED), (MAIN, SINGLE), ("ensemble_7", PUBLISHED), ("ensemble_7", SINGLE), ("ensemble_7", MAIN),
         (SINGLE, PUBLISHED)]                               # the last pair is the report's primary comparison (a check)


def averaged(members: list, key: str) -> np.ndarray:
    """Mean of the members' probabilities (each with its own temperature), returned as a score (logit)."""
    p = np.mean([1.0 / (1.0 + np.exp(-m[key] / float(m["cfg"]["temperature"]))) for m in members], axis=0)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def build() -> dict:
    v1, v2 = R.load_v1(), R.load_v2()
    models = {PUBLISHED: dict(v1[PUBLISHED], temperature=float(v1[PUBLISHED]["cfg"].get("temperature", 1.0))),
              SINGLE: dict(v2[SINGLE], temperature=float(v2[SINGLE]["cfg"]["temperature"]))}
    for name, members in ENSEMBLES.items():
        missing = [m for m in members if m not in v2]
        if missing:
            sys.exit(f"{name}: no stored scores for {missing}")
        models[name] = dict(kind="exploratory ensemble", members=members, temperature=1.0,   # already a mean of probabilities
                            rsna=averaged([v2[m] for m in members], "rsna"), kermany=averaged([v2[m] for m in members], "kermany"))
    return models


def cell(v, c, d=3):
    return f"{v:.{d}f} ({c[0]:.{d}f} to {c[1]:.{d}f})"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["val", "test"])
    ap.add_argument("--again", default=None, metavar="REASON", help="compute the test numbers a second time; the reason is stored")
    args = ap.parse_args()

    if not R.committed(ADDENDUM):
        sys.exit("docs/V2_ADDENDUM_ENSEMBLE.md is not committed. The rules must be in git before any number is computed.")
    _, mk = data.load_kermany()
    _, mr = data.load_rsna()
    yr, yk = mr.label.values.astype(int), mk.label.values.astype(int)
    adult = mr.is_adult.values
    val, test = (mr.split == "val").values & adult, (mr.split == "test").values & adult
    kval, ktest = (mk.split == "val").values, (mk.split == "test").values
    is_pa, is_ap = (mr["view"] == "PA").values, (mr["view"] == "AP").values
    models = build()
    names = list(models)
    OUT.mkdir(parents=True, exist_ok=True)

    if args.what == "val":
        rows = {n: dict(adult_val_auroc=metrics.auroc(yr[val], m["rsna"][val]),
                        adult_val_auroc_PA=metrics.auroc(yr[val & is_pa], m["rsna"][val & is_pa]),
                        adult_val_auroc_AP=metrics.auroc(yr[val & is_ap], m["rsna"][val & is_ap]),
                        children_val_auroc=metrics.auroc(yk[kval], m["kermany"][kval]), members=m.get("members"))
                for n, m in models.items()}
        (OUT / "ensemble_validation.json").write_text(json.dumps(dict(
            note="Validation X-rays only. Written before any test number of an averaged model is computed. "
                 "The members are fixed by name in docs/V2_ADDENDUM_ENSEMBLE.md; nothing is chosen from this table.", rows=rows), indent=1))
        for n, r in rows.items():
            print(f"{n:18s} adult validation AUROC {r['adult_val_auroc']:.4f}  PA {r['adult_val_auroc_PA']:.4f}  "
                  f"AP {r['adult_val_auroc_AP']:.4f}  children {r['children_val_auroc']:.4f}")
        print("\nwrote results/exploratory/ensemble_validation.json. Commit it, then run: python scripts/13_ensemble_check.py test")
        sys.exit(0)

    if not R.committed(OUT / "ensemble_validation.json"):
        sys.exit("results/exploratory/ensemble_validation.json is not committed. Run 'val', commit, then 'test'.")
    if (OUT / "ensemble.json").exists() and not args.again:
        sys.exit("results/exploratory/ensemble.json exists: the test numbers were already computed. The addendum says once.")

    patients = mr.nih_patient.astype(str).values
    normal = (mr.cls == "Normal").values
    groups = {"adults": test, "adults_PA": test & is_pa, "adults_AP": test & is_ap}
    S = np.stack([models[n]["rsna"] for n in names])
    thr = lambda s, rows: dict(A=R.thr_sensitivity(yr[rows], s[rows]), C=R.thr_specificity(yr[rows], s[rows]))  # noqa: E731
    val_rows = np.flatnonzero(val)
    res = {}
    for n in names:
        m, s = models[n], models[n]["rsna"]
        t = thr(s, val_rows)
        sig = lambda z, T: 1.0 / (1.0 + np.exp(-z / T))  # noqa: E731
        res[n] = dict(kind=m["kind"], members=m.get("members"),
                      auroc={g: dict(value=float(R.auroc_rows(s[None, mask], yr[mask])[0])) for g, mask in groups.items()},
                      rules={rule: dict(threshold=t[rule], **{g: R.rates(yr[mask], s[mask], t[rule]) for g, mask in groups.items()}) for rule in t},
                      children=dict(auroc=metrics.auroc(yk[ktest], m["kermany"][ktest]), at_point_A=R.rates(yk[ktest], m["kermany"][ktest], t["A"])),
                      calibration_error_adults=metrics.ece(yr[test], sig(s[test], m["temperature"])),
                      calibration_error_adults_temperature_from_adult_validation=metrics.ece(yr[test], sig(s[test], fit_temperature(s[val], yr[val]))),
                      view_shortcut_auroc_among_normal_adults=float(R.auroc_rows(s[None, test & normal], is_ap[test & normal].astype(int))[0]))

    # the same patient resamples as scripts/10_v2_report.py: same seed, same order of draws
    rng = np.random.default_rng(R.SEED)
    sampler, val_sampler = R.PatientSampler(patients, test), R.PatientSampler(patients, val)
    boot = {g: np.full((R.N_BOOT, len(names)), np.nan) for g in groups}
    boot_rule = {(n, r, g, k): [] for n in names for r in "AC" for g in ("adults", "adults_PA") for k in ("sensitivity", "specificity")}
    for b in range(R.N_BOOT):
        idx = sampler.draw(rng)
        boot["adults"][b] = R.auroc_rows(S[:, idx], yr[idx])
        for g, mask in (("adults_PA", is_pa), ("adults_AP", is_ap)):
            sub = idx[mask[idx]]
            boot[g][b] = R.auroc_rows(S[:, sub], yr[sub])
        vidx = val_sampler.draw(rng)
        pa = idx[is_pa[idx]]
        for n in names:
            s = models[n]["rsna"]
            t = thr(s, vidx)
            for rule in "AC":
                for g, rows in (("adults", idx), ("adults_PA", pa)):
                    r = R.rates(yr[rows], s[rows], t[rule])
                    boot_rule[(n, rule, g, "sensitivity")].append(r["sensitivity"])
                    boot_rule[(n, rule, g, "specificity")].append(r["specificity"])
    for i, n in enumerate(names):
        for g in groups:
            res[n]["auroc"][g]["ci"] = R.ci(boot[g][:, i])
        for rule in "AC":
            for g in ("adults", "adults_PA"):
                for k in ("sensitivity", "specificity"):
                    res[n]["rules"][rule][g][k + "_ci"] = R.ci(boot_rule[(n, rule, g, k)])
    paired = {}
    for a, b in PAIRS:
        ia, ib = names.index(a), names.index(b)
        paired[f"{a} minus {b}"] = {g: dict(difference=res[a]["auroc"][g]["value"] - res[b]["auroc"][g]["value"],
                                            ci=R.ci(boot[g][:, ia] - boot[g][:, ib])) for g in groups}

    # check against the report: the primary comparison must come out identical
    ref = json.loads((C.RESULTS_DIR / "v2_report.json").read_text())["primary_comparison"]
    got = paired[f"{SINGLE} minus {PUBLISHED}"]["adults"]
    if abs(got["difference"] - ref["difference"]) > 1e-9 or max(abs(np.array(got["ci"]) - np.array(ref["ci"]))) > 1e-9:
        sys.exit(f"CHECK FAILED: this script gets {got} for the report's primary comparison, the report says {ref['difference']}, {ref['ci']}")

    d = paired[f"{MAIN} minus {PUBLISHED}"]
    better = bool(d["adults"]["ci"][0] > 0 and d["adults_PA"]["difference"] > 0 and d["adults_AP"]["difference"] > 0)
    report = dict(
        status="EXPLORATORY: added on 15 Nov 2026 after the test results of the single models were seen (docs/V2_ADDENDUM_ENSEMBLE.md). "
               "It does not replace the primary comparison of docs/V2_PROTOCOL.md.",
        how="mean of the members' probabilities (each with its own temperature); no training, no weights, members fixed by name",
        ensembles=ENSEMBLES, models=res, paired_auroc_differences=paired,
        main_comparison=dict(what=f"adult test AUROC, {MAIN} minus published {PUBLISHED}, same {R.N_BOOT} patient resamples as the report",
                             rule="called better only if the overall interval lies above zero and the difference is positive inside both views",
                             better=better, **d),
        check="the report's primary comparison was recomputed here and is identical",
        computed_at_utc=time.strftime("%Y-%m-%d %H:%M", time.gmtime()), computed_again_because=args.again)
    (OUT / "ensemble.json").write_text(json.dumps(report, indent=1))

    n_t, n_pos = int(test.sum()), int(yr[test].sum())
    L = ["# Averaging the networks we already trained (written by scripts/13_ensemble_check.py)", "",
         "**Exploratory.** Rules: docs/V2_ADDENDUM_ENSEMBLE.md, committed before any of these numbers was computed. Added after the test",
         "results of the single models were seen, so this does not replace the verdict of the protocol's primary comparison.",
         "No training: the scores of finished runs are averaged. 'Lung opacity' is a radiologist's finding, not a confirmed pneumonia.", "",
         f"## Adults, lung opacity vs everything else (n = {n_t:,}, {100 * n_pos / n_t:.1f}% with lung opacity)", "",
         "| Model | Networks | AUROC | inside standing (PA) | inside bedside (AP) | A: sens / spec | A: right answers | C: sens / spec |", "|---|---|---|---|---|---|---|---|"]
    for n in names:
        r, a = res[n], res[n]["auroc"]
        pr = lambda rule: f"{r['rules'][rule]['adults']['sensitivity']:.2f} / {r['rules'][rule]['adults']['specificity']:.2f}"  # noqa: E731
        L.append(f"| {n} | {len(r['members'] or [n])} | {cell(a['adults']['value'], a['adults']['ci'])} | {a['adults_PA']['value']:.3f} | "
                 f"{a['adults_AP']['value']:.3f} | {pr('A')} | {r['rules']['A']['adults']['accuracy']:.3f} | {pr('C')} |")
    L += ["", "## Paired differences in adult AUROC (same patient resamples as results/v2_report.md)", "",
          "| Comparison | Adults | inside PA | inside AP |", "|---|---|---|---|"]
    for k, p in paired.items():
        c = lambda g: f"{p[g]['difference']:+.3f} ({p[g]['ci'][0]:+.3f} to {p[g]['ci'][1]:+.3f})"  # noqa: E731
        L.append(f"| {k} | {c('adults')} | {c('adults_PA')} | {c('adults_AP')} |")
    L += ["", f"Rule of the protocol applied to {MAIN} minus {PUBLISHED}: {report['main_comparison']['rule']}. "
          f"**Result: {'the averaged model is better (exploratory).' if better else 'no improvement is shown.'}**", "",
          "## Points A and C with intervals, standing X-rays, children, further checks", "",
          "| Model | A: sensitivity | A: specificity | C: sensitivity | C: specificity | PA at A: sens / spec | Children: AUROC | Children at A: sens / spec | "
          "Calibration error, adults | Bedside vs standing among Normal adults |", "|---|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        r = res[n]
        ra, rc, pa, ch = r["rules"]["A"]["adults"], r["rules"]["C"]["adults"], r["rules"]["A"]["adults_PA"], r["children"]
        L.append(f"| {n} | {cell(ra['sensitivity'], ra['sensitivity_ci'], 2)} | {cell(ra['specificity'], ra['specificity_ci'], 2)} | "
                 f"{cell(rc['sensitivity'], rc['sensitivity_ci'], 2)} | {cell(rc['specificity'], rc['specificity_ci'], 2)} | "
                 f"{pa['sensitivity']:.2f} / {pa['specificity']:.2f} | {ch['auroc']:.3f} | {ch['at_point_A']['sensitivity']:.2f} / {ch['at_point_A']['specificity']:.2f} | "
                 f"{r['calibration_error_adults']:.3f} | {r['view_shortcut_auroc_among_normal_adults']:.2f} |")
    (OUT / "ensemble.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
