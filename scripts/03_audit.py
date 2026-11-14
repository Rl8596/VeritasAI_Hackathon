"""Step 3: audit both datasets before trusting any model number.

    python scripts/03_audit.py

A1  Who is in the data?          counts per split and class; age, sex, view and opacity rate in RSNA
A2  Leakage                      identical and near-identical images within and across splits and datasets;
                                 for RSNA also: the same patient showing up in several splits
A3  Shortcuts, children          can the label be guessed from image size, brightness or the image border?
A4  Shortcuts, adults            can "lung opacity" be guessed from view (AP/PA), age and sex?
A5  "Which hospital is this?"    can a simple classifier tell the two datasets apart, even at the same age?

Nothing here needs the fine-tuned model. Every classifier in A3 is fitted on Kermany train, choices are
made on Kermany val, and test data is only scored. Outputs: results/audit.json, results/figures/audit_*.png.
Temporary files go to PNEUMOSHIFT_SCRATCH (default: <system temp dir>/pneumoshift_scratch).
One CPU thread by default (set OMP_NUM_THREADS to use more); about 6 minutes.
"""
import os

THREADS = int(os.environ.get("OMP_NUM_THREADS", "1"))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = str(THREADS)

import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import urllib.request  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pneumoshift import config as C, data, metrics, plotting  # noqa: E402
from pneumoshift.preprocess import center_crop  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402  (after plotting, which selects the Agg backend)
from matplotlib.patches import Patch  # noqa: E402

torch.set_num_threads(THREADS)
plotting.use_style()

SCRATCH = Path(os.environ.get("PNEUMOSHIFT_SCRATCH", Path(tempfile.gettempdir()) / "pneumoshift_scratch"))
SCRATCH.mkdir(parents=True, exist_ok=True)
C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Near-duplicate search. Thresholds were set by LOOKING at the pairs (see audit_duplicate_pairs.png):
# every pair at or above 0.98 is a pixel-identical copy, the most similar non-identical pairs
# (0.90-0.98) are different exposures or different patients.
CANDIDATE_CORR_32 = 0.90     # stage 1: correlation of 32x32 thumbnails, all pairs
NEAR_DUP_CORR_256 = 0.98     # stage 2: correlation of the full 256 px images
SIMILAR_CORR_256 = 0.95      # reported separately as "very similar, not the same image"
# RSNA "patientId" is really an image id. RSNA published the link back to the NIH ChestX-ray14 file
# names, whose first 8 digits are the true patient number.
NIH_MAPPING_URL = "https://s3.amazonaws.com/east1.public.rsna.org/AI/2018/pneumonia-challenge-dataset-mappings_2018.json"
PERCENTILES = [1, 5, 25, 50, 75, 95, 99]
AGE_BANDS = [(1, 5), (6, 17), (18, 29), (30, 39), (40, 49), (50, 59), (60, 69), (70, 100)]
NOT_NORMAL = "No Lung Opacity / Not Normal"
KEY_CHILD = "children_test (Kermany, pneumonia vs normal)"
KEY_ADULT_NORMAL = "adults_test (RSNA, lung opacity vs normal)"
KEY_ADULT_ALL = "adults_test (RSNA, lung opacity vs everything else)"

T0 = time.time()
popcount = getattr(np, "bitwise_count", None) or (
    lambda a: np.unpackbits(np.ascontiguousarray(a)[..., None].view(np.uint8), axis=-1).sum(-1))


def log(msg):
    print(f"[{time.time() - T0:5.0f}s] {msg}", flush=True)


def clean(o):
    """Make numpy types and NaN safe for strict JSON."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else round(float(o), 6)
    if isinstance(o, (np.integer, np.bool_)):
        return o.item()
    return o


def auroc_ci(y, p):
    lo, hi = metrics.bootstrap_ci(metrics.auroc, y, p)
    return dict(auroc=metrics.auroc(y, p), auroc_ci=[lo, hi], n=int(len(y)), n_pos=int(np.sum(y)))


# --------------------------------------------------------------------------
# Load, and one pass over both arrays for every per-image quantity the audit needs
# --------------------------------------------------------------------------
Xk, mk = data.load_kermany()
Xr, mr = data.load_rsna()
nk, nr = len(mk), len(mr)
yk, yr = mk.label.values, mr.label.values
k_train, k_val, k_test = [(mk.split == s).values for s in ("train", "val", "test")]
adult = mr.is_adult.values
r_adapt, r_val, r_test = [((mr.split == s) & mr.is_adult).values for s in ("adapt", "val", "test")]
vs_normal = (mr.cls != NOT_NORMAL).values


def frame_blocks(a, width=16):
    """Outer `width`-pixel frame of square images [m, S, S], as the mean of each 16x16 block on the frame."""
    m, s = a.shape[0], a.shape[1] // width
    b = a.reshape(m, s, width, s, width).mean((2, 4))
    ring = np.ones((s, s), bool)
    ring[1:-1, 1:-1] = False
    return b[:, ring] / 255.0


def scan(X):
    n = len(X)
    f = dict(sha=np.empty(n, "U40"), dhash=np.zeros(n, np.uint64),
             thumb256=np.zeros((n, 1024), np.float32),      # whole stored image, 32x32 (duplicate search)
             thumb224=np.zeros((n, 1024), np.float32),      # the 224 px centre crop the network sees, 32x32
             bright=np.zeros((n, 2 + len(PERCENTILES)), np.float32),
             border_outer=np.zeros((n, 60), np.float32),    # frame of the 256 px image (cut off at test time)
             border_seen=np.zeros((n, 52), np.float32))     # frame of the 224 px crop (the network sees it)
    bit = np.uint64(1) << np.arange(64, dtype=np.uint64)
    for s in range(0, n, 512):
        a = np.asarray(X[s:s + 512])
        m, sl = len(a), slice(s, s + len(a))
        c = center_crop(a, C.IMG_SIZE).astype(np.float32)
        flat = c.reshape(m, -1)
        f["thumb256"][sl] = a.reshape(m, 32, 8, 32, 8).mean((2, 4)).reshape(m, -1)
        f["thumb224"][sl] = c.reshape(m, 32, 7, 32, 7).mean((2, 4)).reshape(m, -1) / 255.0
        f["bright"][sl] = np.c_[flat.mean(1), flat.std(1), np.percentile(flat, PERCENTILES, axis=1).T] / 255.0
        f["border_outer"][sl] = frame_blocks(a)
        f["border_seen"][sl] = frame_blocks(c)
        for j, im in enumerate(a):
            f["sha"][s + j] = hashlib.sha1(im.tobytes()).hexdigest()
            g = np.asarray(Image.fromarray(im).resize((9, 8), Image.Resampling.BOX), dtype=np.int16)
            f["dhash"][s + j] = bit[(g[:, 1:] > g[:, :-1]).ravel()].sum(dtype=np.uint64)   # 64-bit dHash
    return f


fk = scan(Xk)
log(f"scanned Kermany ({nk} images)")
fr = scan(Xr)
log(f"scanned RSNA ({nr} images)")
audit = {}

# ==========================================================================
# A1. Who is in the data?
# ==========================================================================
log("A1 populations")


def by_split(df, col, order, names=None):
    t = df.groupby(["split", col]).size().unstack(fill_value=0)
    out = {}
    for s in order:
        row = {str(names[c] if names else c): int(t.loc[s, c]) for c in t.columns}
        row["total"] = int(t.loc[s].sum())
        out[s] = row
    return out


def rate_table(df, key):
    """Lung-opacity rate per subgroup, both against everything else and against 'Normal' only."""
    out = {}
    for k, g in df.groupby(key, observed=True, sort=False):
        n, n_op, n_norm = len(g), int(g.label.sum()), int((g.cls == "Normal").sum())
        out[str(k)] = dict(n=n, n_lung_opacity=n_op, n_normal=n_norm, n_not_normal_no_opacity=n - n_op - n_norm,
                           opacity_rate=n_op / n,
                           opacity_rate_vs_normal_only=n_op / (n_op + n_norm) if n_op + n_norm else float("nan"))
    return out


def age_band(age):
    out = pd.Series("invalid", index=age.index)
    for lo, hi in AGE_BANDS:
        out[age.between(lo, hi)] = f"{lo}-{hi}" if hi < 100 else f"{lo}+"
    return pd.Categorical(out, [f"{lo}-{hi}" if hi < 100 else f"{lo}+" for lo, hi in AGE_BANDS] + ["invalid"])


mr["age_band"] = age_band(mr.age)
valid_age = mr.age[mr.age_valid]
k_cls = by_split(mk, "label", ["train", "val", "test"], names={0: "normal", 1: "pneumonia"})
for s, row in k_cls.items():
    row["pneumonia_share"] = row["pneumonia"] / row["total"]
size_by_label = {name: {c: float(mk[c][k_train & (yk == lab)].median()) for c in ("orig_w", "orig_h")}
                 | {"aspect_w_over_h": float((mk.orig_w / mk.orig_h)[k_train & (yk == lab)].median())}
                 for lab, name in ((0, "normal"), (1, "pneumonia"))}
r_tables = {}
for pop_name, pop in (("all_images_with_valid_age", mr[mr.age_valid]), ("adults", mr[adult]), ("adult_test_set", mr[r_test])):
    r_tables[pop_name] = dict(n=int(len(pop)), by_view=rate_table(pop, "view"), by_sex=rate_table(pop, "sex"),
                              by_age_band=rate_table(pop, "age_band"))
audit["A1_who_is_in_the_data"] = dict(
    kermany_children=dict(
        n_images=nk, age="1-5 years for every patient according to Kermany et al. 2018; the files carry no per-image age, sex or view",
        by_split_and_class=k_cls,
        by_split_and_subtype=by_split(mk, "subtype", ["train", "val", "test"]),
        patient_groups_per_split={s: int(mk.patient_id[mk.split == s].nunique()) for s in ("train", "val", "test")},
        official_split_sizes={str(k): int(v) for k, v in mk.official_split.value_counts().items()},
        median_original_image_size_in_train=size_by_label),
    rsna_mostly_adults=dict(
        n_images=nr,
        by_split_and_class=by_split(mr, "cls", ["adapt", "val", "test"]),
        adults_by_split_and_class=by_split(mr[adult], "cls", ["adapt", "val", "test"]),
        age=dict(n_valid=int(mr.age_valid.sum()), invalid_values_removed=sorted(int(a) for a in mr.age[~mr.age_valid]),
                 min=int(valid_age.min()), p25=float(valid_age.quantile(.25)), median=float(valid_age.median()),
                 mean=float(valid_age.mean()), p75=float(valid_age.quantile(.75)), max=int(valid_age.max()),
                 n_adults_18_plus=int(adult.sum()), n_under_18=int((valid_age < 18).sum()),
                 n_age_1_to_5=int((valid_age <= 5).sum()),
                 share_age_1_to_5=float((valid_age <= 5).mean()),
                 counts_by_age_band={str(k): int(v) for k, v in mr.age_band.value_counts(sort=False).items()}),
        sex={str(k): int(v) for k, v in mr.sex.value_counts().items()},
        view={str(k): int(v) for k, v in mr["view"].value_counts().items()},
        view_by_sex_adults={f"{v}_{s}": int(((mr["view"] == v) & (mr.sex == s) & adult).sum())
                            for v in ("AP", "PA") for s in ("F", "M")},
        lung_opacity_rate=r_tables))

# Figure: age of the two populations
fig, ax = plt.subplots(figsize=(10, 5))
ax.axvspan(0.5, 5.5, color=plotting.CHILDREN, alpha=0.35, lw=0, zorder=1)
ax.hist(valid_age, bins=np.arange(0.5, valid_age.max() + 1.5, 1), color=plotting.ADULTS, rwidth=0.86, zorder=3)
ax.set_xlim(0, valid_age.max() + 2)
ax.set_xticks(range(0, int(valid_age.max()) + 2, 10))
ax.set_xlabel("Age of the patient (years)")
ax.set_ylabel("Number of X-rays")
ax.grid(axis="x", visible=False)
ax.legend(handles=[
    Patch(color=plotting.CHILDREN, alpha=0.35, label=f"Children's dataset (Kermany, Guangzhou): all {nk:,} X-rays are from ages 1–5"),
    Patch(color=plotting.ADULTS, label=f"Adult dataset (RSNA, USA): {len(valid_age):,} X-rays, median age {valid_age.median():.0f}")],
    loc="upper right", fontsize=11)
ax.set_ylim(0, ax.get_ylim()[1] * 1.22)
ax.set_title(f"The two datasets barely overlap in age: only {int((valid_age <= 5).sum())} of {len(valid_age):,} "
             "US X-rays are from 1–5-year-olds", pad=12)
plotting.finish(fig, C.FIGURES_DIR / "audit_age_populations.png",
                note="Kermany has no per-image ages; the 1–5 range is from the paper (Kermany et al. 2018), so it is shown as a band, "
                     f"not as counts.\nRSNA ages are from the DICOM headers; {int((~mr.age_valid).sum())} impossible ages (over 100) removed.")

# Figure: lung-opacity rate by view, age and sex among RSNA adults
ad = r_tables["adults"]
panels = [("How the X-ray was taken", [("AP", "AP: front-to-back,\nusually bedside / portable"), ("PA", "PA: back-to-front,\nstanding")], ad["by_view"]),
          ("Age", [(b, f"{b} years") for b in ad["by_age_band"]], ad["by_age_band"]),
          ("Sex", [("F", "Women"), ("M", "Men")], ad["by_sex"])]
fig, axes = plt.subplots(3, 1, figsize=(10, 7.6), sharex=True, gridspec_kw=dict(height_ratios=[2, len(panels[1][1]), 2], hspace=0.42))
for ax, (title, rows, tab) in zip(axes, panels):
    vals = [100 * tab[k]["opacity_rate"] for k, _ in rows]
    ax.barh(range(len(rows)), vals, height=0.62, color=plotting.ADULTS, zorder=3)
    for i, (k, _) in enumerate(rows):
        ax.annotate(f"{vals[i]:.0f}%  ({tab[k]['n_lung_opacity']:,} of {tab[k]['n']:,})", (vals[i], i), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=11, color=plotting.TEXT)
    ax.set_yticks(range(len(rows)), [lab for _, lab in rows])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, fontsize=12, fontweight="normal", color=plotting.TEXT_2)
axes[-1].set_xlim(0, 62)
axes[-1].set_xticks([0, 10, 20, 30, 40, 50], ["0%", "10%", "20%", "30%", "40%", "50%"])
axes[-1].set_xlabel("Share of X-rays labelled \"lung opacity\" (possible pneumonia)")
fig.suptitle(f"In the adult dataset the label follows how the X-ray was taken:\n"
             f"{100 * ad['by_view']['AP']['opacity_rate']:.0f}% of bedside (AP) X-rays show lung opacity, "
             f"but only {100 * ad['by_view']['PA']['opacity_rate']:.0f}% of standing (PA) ones",
             x=0.01, ha="left", fontsize=14, fontweight="bold", y=1.0)
plotting.finish(fig, C.FIGURES_DIR / "audit_adult_opacity_rate.png",
                note=f"RSNA adults (18+), all splits, n={ad['n']:,}. Rate = lung opacity vs everything else.")

# ==========================================================================
# A2. Leakage: identical and near-identical images
# ==========================================================================
log("A2 leakage: all-pairs search")
n = nk + nr
sha = np.concatenate([fk["sha"], fr["sha"]])
dh = np.concatenate([fk["dhash"], fr["dhash"]])
is_rsna = np.arange(n) >= nk
g_split = np.concatenate([mk.split.values.astype(str), mr.split.values.astype(str)])
Z = np.concatenate([fk["thumb256"], fr["thumb256"]])
Z -= Z.mean(1, keepdims=True)
Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8

# Stage 1: every one of the n*(n-1)/2 pairs: dHash distance (for the record) and thumbnail correlation.
cols = np.arange(n)
ham_hist = np.zeros(3 * 65, np.int64)                  # kind (0 = K-K, 1 = K-R, 2 = R-R) x Hamming distance
cand = []
for s in range(0, n, 1000):
    rows = np.arange(s, min(s + 1000, n))
    c = Z[rows] @ Z.T
    h = popcount(dh[rows, None] ^ dh[None, :])
    upper = rows[:, None] < cols[None, :]
    code = h.astype(np.int16) + 65 * (is_rsna[rows, None].astype(np.int16) + is_rsna[None, :])
    ham_hist += np.bincount(code[upper], minlength=3 * 65)
    ii, jj = np.nonzero((c >= CANDIDATE_CORR_32) & upper)
    cand.append(np.c_[rows[ii], jj, c[ii, jj], h[ii, jj]])
pairs = pd.DataFrame(np.concatenate(cand), columns=["i", "j", "c32", "ham"]).astype({"i": int, "j": int, "ham": int})
ham_hist = ham_hist.reshape(3, 65)
log(f"   {n * (n - 1) // 2:,} pairs compared, {len(pairs):,} candidates with thumbnail correlation >= {CANDIDATE_CORR_32}")


def image(g):
    return np.asarray(Xk[g] if g < nk else Xr[g - nk])


# Stage 2: correlation of the full 256 px images for every candidate pair.
_cache = {}


def unit(g):
    if g not in _cache:
        a = image(g).astype(np.float32).ravel()
        a -= a.mean()
        if len(_cache) > 4000:
            _cache.pop(next(iter(_cache)))
        _cache[g] = a / (np.linalg.norm(a) + 1e-8)
    return _cache[g]


pairs = pairs.sort_values(["i", "j"]).reset_index(drop=True)
pairs["c256"] = [float(unit(i) @ unit(j)) for i, j in zip(pairs.i, pairs.j)]
_cache.clear()
pairs["exact"] = sha[pairs.i.values] == sha[pairs.j.values]
log("   candidates verified at full resolution")


def category(i, j):
    a, b = g_split[i], g_split[j]
    if not is_rsna[i] and not is_rsna[j]:
        if a == b:
            return f"within kermany {a}"
        return "kermany train+val vs kermany test" if "test" in (a, b) else "kermany train vs kermany val"
    if is_rsna[i] != is_rsna[j]:
        return "kermany vs rsna"
    if a == b:
        return f"within rsna {a}"
    a, b = sorted((a, b), key=["adapt", "val", "test"].index)
    return f"rsna {a} vs rsna {b}"


pairs["category"] = [category(i, j) for i, j in zip(pairs.i, pairs.j)]
CATEGORIES = ["kermany train+val vs kermany test", "kermany train vs kermany val", "within kermany train", "within kermany val",
              "within kermany test", "kermany vs rsna", "rsna adapt vs rsna test", "rsna adapt vs rsna val",
              "rsna val vs rsna test", "within rsna adapt", "within rsna val", "within rsna test"]
by_category = {}
for cat in CATEGORIES:
    p = pairs[pairs.category == cat]
    nonexact = p[~p.exact]
    by_category[cat] = dict(
        identical_pairs=int(p.exact.sum()),
        near_duplicate_pairs_not_identical=int((nonexact.c256 >= NEAR_DUP_CORR_256).sum()),
        very_similar_pairs_0p95_to_0p98=int(nonexact.c256.between(SIMILAR_CORR_256, NEAR_DUP_CORR_256, inclusive="left").sum()),
        highest_correlation_of_a_non_identical_pair=float(nonexact.c256.max()) if len(nonexact) else None)

# Identical images, as groups
sha_s = pd.Series(sha)
dup_groups = [np.array(ix) for ix in sha_s[sha_s.duplicated(keep=False)].groupby(sha_s).groups.values()]
k_groups = [g for g in dup_groups if g.max() < nk]
r_groups = [g for g in dup_groups if g.min() >= nk]


def describe_k_group(g):
    rws = mk.iloc[g]
    return dict(files=rws.filename.tolist(), splits=sorted(set(rws.split)), labels=sorted(set(int(v) for v in rws.label)),
                patient_ids=sorted(set(rws.patient_id)))


k_desc = [describe_k_group(g) for g in k_groups]
r_desc = [dict(image_ids=mr.patient_id.iloc[g - nk].tolist(), splits=sorted(set(mr.split.iloc[g - nk])),
               classes=sorted(set(mr.cls.iloc[g - nk])), age_sex_view=[f"{a}{s} {v}" for a, s, v in
                                                                      zip(mr.age.iloc[g - nk], mr.sex.iloc[g - nk], mr["view"].iloc[g - nk])])
          for g in r_groups]
identical = dict(
    how="SHA-1 of the stored 256x256 array",
    kermany=dict(groups=len(k_groups), images_in_groups=int(sum(len(g) for g in k_groups)),
                 redundant_images=int(sum(len(g) - 1 for g in k_groups)),
                 redundant_images_by_split={s: int(sum(len(g) - 1 for g, d in zip(k_groups, k_desc) if d["splits"] == [s]))
                                            for s in ("train", "val", "test")},
                 groups_spanning_two_splits=int(sum(len(d["splits"]) > 1 for d in k_desc)),
                 groups_with_conflicting_labels=int(sum(len(d["labels"]) > 1 for d in k_desc)),
                 groups_with_two_different_patient_ids=int(sum(len(d["patient_ids"]) > 1 for d in k_desc)),
                 groups_list=k_desc),
    rsna=dict(groups=len(r_groups), images_in_groups=int(sum(len(g) for g in r_groups)),
              groups_spanning_two_splits=int(sum(len(d["splits"]) > 1 for d in r_desc)), groups_list=r_desc),
    kermany_vs_rsna=dict(groups=len(dup_groups) - len(k_groups) - len(r_groups)))

# How useful was dHash on its own? (It is not: chest X-rays of different people share the same coarse layout.)
kinds = ["kermany-kermany", "kermany-rsna", "rsna-rsna"]
pairs["kind"] = is_rsna[pairs.i.values].astype(int) + is_rsna[pairs.j.values]
confirmed = pairs[pairs.c256 >= NEAR_DUP_CORR_256]
dhash_report = {}
for t in (0, 2, 5, 10):
    dhash_report[f"hamming_distance_le_{t}"] = {
        kinds[k]: dict(pairs_flagged=int(ham_hist[k, :t + 1].sum()),
                       of_which_confirmed_same_image=int(((confirmed.kind == k) & (confirmed.ham <= t)).sum()))
        for k in range(3)}
dhash_report["confirmed_same_image_pairs_total"] = {kinds[k]: int((confirmed.kind == k).sum()) for k in range(3)}
dhash_report["largest_hamming_distance_among_confirmed_pairs"] = int(confirmed.ham.max()) if len(confirmed) else None


# ---- Kermany: are the patient ids parsed from the file names right, and is the official test set separate?
series = np.where(yk == 1, mk.subtype.values.astype(str),
                  np.where(mk.filename.str.startswith("NORMAL2").values, "NORMAL2-IM", "IM"))
number = mk.patient_id.str.extract(r"_[PN](\d+)$")[0].astype(int).values
official_test = (mk.official_split == "test").values
numbering = {}
for sname in ("bacteria", "virus", "IM", "NORMAL2-IM"):
    m = series == sname
    a, b = set(number[m & ~official_test]), set(number[m & official_test])
    numbering[sname] = dict(patient_numbers_in_official_train_val=len(a), patient_numbers_in_official_test=len(b),
                            numbers_in_both=len(a & b))
Zk = Z[:nk]
Ck = Zk @ Zk.T
upper_k = np.triu(np.ones((nk, nk), bool), 1)
same_number = number[:, None] == number[None, :]
same_series = series[:, None] == series[None, :]
same_label = yk[:, None] == yk[None, :]
similarity = {
    "same series, same number (same patient)": float(Ck[upper_k & same_number & same_series].mean()),
    "same number, other series (bacteria N vs virus N, or IM-N vs NORMAL2-IM-N)": float(Ck[upper_k & same_number & ~same_series & same_label].mean()),
    "different number, same class": float(Ck[upper_k & ~same_number & same_label].mean())}
fine_id = pd.Series(series).str.cat(pd.Series(number).astype(str), sep="_")
kermany_ids = dict(
    patient_groups_used_for_the_split=int(mk.patient_id.nunique()),
    distinct_series_and_number_combinations=int(fine_id.nunique()),
    groups_that_merge_two_numbering_series=int((fine_id.groupby(mk.patient_id.values).nunique() > 1).sum()),
    numbering_per_series=numbering,
    mean_thumbnail_correlation=similarity,
    reading="Within each series (bacteria, virus, IM, NORMAL2-IM) no patient number occurs in both the official "
            "train/val folders and the official test folder, so the official test set is a separate block of patients. "
            "data.py groups by the number alone, which merges e.g. 'person12_bacteria' and 'person12_virus'. Their images "
            "are no more alike than images of unrelated children, so these are different patients: the grouping is "
            "stricter than needed (safe for the split), but it under-counts patients.")
del Ck, same_number, same_series, same_label, upper_k


# ---- RSNA: the same patient in several splits (needs RSNA's own mapping back to the NIH file names)
def load_nih_mapping():
    f = SCRATCH / "nih_mapping" / "pneumonia-challenge-dataset-mappings_2018.json"
    if not f.exists():
        try:
            f.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(NIH_MAPPING_URL, timeout=120) as resp:
                raw = resp.read()
            f.write_bytes(raw)
            log(f"   downloaded the RSNA -> NIH mapping ({len(raw) / 1e6:.0f} MB)")
        except Exception as e:  # no internet: the audit still runs, this check is skipped
            log(f"   could not download the RSNA -> NIH mapping ({type(e).__name__}: {e})")
            return None
    m = pd.DataFrame(json.loads(f.read_text()))[["subset_img_id", "img_id"]]
    return mr[["patient_id"]].merge(m, left_on="patient_id", right_on="subset_img_id", how="left").img_id.str.slice(0, 8)


nih = load_nih_mapping()
if nih is not None and nih.notna().all():
    nihv = nih.values.astype(str)
    per_patient = pd.Series(nihv).value_counts()
    sets = {s: set(nihv[(mr.split == s).values]) for s in ("adapt", "val", "test")}
    seen_in_adapt = np.isin(nihv, list(sets["adapt"]))
    seen_in_adapt_or_val = np.isin(nihv, list(sets["adapt"] | sets["val"]))
    clean_test = r_test & ~seen_in_adapt_or_val
    rsna_overlap = dict(
        checked=True, source=NIH_MAPPING_URL,
        how="RSNA 'patientId' is unique per IMAGE. RSNA's mapping file gives the NIH ChestX-ray14 file name of every "
            "image (e.g. 00012773_000.png); the first 8 digits are the real patient number.",
        n_images=nr, n_distinct_patients=int(len(per_patient)),
        patients_with_more_than_one_image=int((per_patient > 1).sum()),
        images_of_patients_with_more_than_one_image=int(per_patient[per_patient > 1].sum()),
        max_images_of_one_patient=int(per_patient.max()),
        sex_identical_within_every_multi_image_patient=bool((mr.groupby(nihv).sex.nunique() == 1).all()),
        patients_in_both={"adapt and test": len(sets["adapt"] & sets["test"]), "adapt and val": len(sets["adapt"] & sets["val"]),
                          "val and test": len(sets["val"] & sets["test"])},
        adult_test_images=dict(
            n=int(r_test.sum()),
            patient_also_in_adapt=int((r_test & seen_in_adapt).sum()),
            share_patient_also_in_adapt=float((r_test & seen_in_adapt).sum() / r_test.sum()),
            patient_also_in_adapt_or_val=int((r_test & seen_in_adapt_or_val).sum()),
            share_patient_also_in_adapt_or_val=float((r_test & seen_in_adapt_or_val).sum() / r_test.sum()),
            share_patient_also_in_adapt_by_class={c: float((r_test & seen_in_adapt & (mr.cls == c).values).sum()
                                                           / (r_test & (mr.cls == c).values).sum()) for c in sorted(mr.cls.unique())},
            distinct_patients=int(len(set(nihv[r_test])))),
        leak_free_adult_test_subset=dict(
            definition="adult test images whose patient has no image in adapt or val",
            n=int(clean_test.sum()), n_lung_opacity=int((clean_test & (yr == 1)).sum()),
            n_normal=int((clean_test & (mr.cls == "Normal").values).sum()),
            n_not_normal_no_opacity=int((clean_test & (mr.cls == NOT_NORMAL).values).sum()),
            file=str(SCRATCH / "rsna_true_patients.csv")),
        consequence="The kids-only model and the baselines never train on RSNA, so their adult numbers are not inflated by this. "
                    "Any model that trains on RSNA 'adapt' and is scored on RSNA 'test' IS affected (same patients on both sides). "
                    "In every case several test images come from the same person, so image-level bootstrap intervals are too narrow.")
    pd.DataFrame(dict(idx=mr.idx, rsna_image_id=mr.patient_id, nih_patient=nihv, split=mr.split,
                      patient_also_in_adapt_or_val=seen_in_adapt_or_val,
                      leak_free_adult_test=clean_test)).to_csv(SCRATCH / "rsna_true_patients.csv", index=False)
    # Does picture similarity agree with true patient identity? (justifies the near-duplicate threshold)
    rr = pairs[(pairs.kind == 2) & ~pairs.exact]
    same_patient = nihv[rr.i.values - nk] == nihv[rr.j.values - nk]
    bands = [(0.90, 0.95), (0.95, 0.97), (0.97, NEAR_DUP_CORR_256), (NEAR_DUP_CORR_256, 1.01)]
    rsna_overlap["similar_looking_rsna_pairs_vs_true_patient"] = {
        f"pixel correlation {lo:.2f}-{min(hi, 1):.2f}": dict(
            pairs=int(rr.c256.between(lo, hi, inclusive="left").sum()),
            share_same_patient=float(same_patient[rr.c256.between(lo, hi, inclusive="left").values].mean())
            if rr.c256.between(lo, hi, inclusive="left").any() else None) for lo, hi in bands}
    pairs["same_nih_patient"] = False
    pairs.loc[pairs.kind == 2, "same_nih_patient"] = (nihv[pairs.i.values[pairs.kind == 2] - nk]
                                                      == nihv[pairs.j.values[pairs.kind == 2] - nk])
else:
    nihv = None
    pairs["same_nih_patient"] = False
    rsna_overlap = dict(checked=False, reason="could not download or match the RSNA -> NIH mapping file", source=NIH_MAPPING_URL)

audit["A2_leakage"] = dict(
    identical_images=identical,
    near_duplicates=dict(
        how=f"Every pair of the {n:,} images was compared with the correlation of 32x32 thumbnails. Pairs at or above "
            f"{CANDIDATE_CORR_32} ({len(pairs):,}) were re-checked with the pixel correlation of the full 256 px images. "
            f"At or above {NEAR_DUP_CORR_256} counts as the same image. The threshold was set by viewing the pairs: all pairs above it "
            "are pixel-identical; the most similar non-identical pairs are different exposures or different people.",
        limits="Finds copies that were re-saved, rescaled or changed in brightness/contrast. Does not find copies that were "
               "cropped or shifted a lot, and cannot recognise a different X-ray of the same patient.",
        candidate_pairs=int(len(pairs)), by_category=by_category),
    dhash_alone_is_not_enough=dict(
        note="64-bit dHash (8x8 gradient hash). 'pairs_flagged' is what dHash alone would have reported; "
             "'of_which_confirmed_same_image' passed the pixel check.", **dhash_report),
    kermany_patient_ids=kermany_ids,
    rsna_same_patient_in_several_splits=rsna_overlap)
log(f"   identical: {len(k_groups)} groups in Kermany, {len(r_groups)} in RSNA; non-identical near-duplicates: "
    f"{sum(v['near_duplicate_pairs_not_identical'] for v in by_category.values())}")

# Figure: example pairs, chosen by rule (not by hand)
examples = []


def first(df):
    return (int(df.i.iloc[0]), int(df.j.iloc[0]), float(df.c256.iloc[0])) if len(df) else None


kk = pairs[pairs.kind == 0]
same_pid = mk.patient_id.values[kk.i.values] == mk.patient_id.values[kk.j.values]
e = first(kk[kk.exact & same_pid & (kk.category == "within kermany train")])
if e:
    examples.append((e, "IDENTICAL image stored twice\n(same child, both copies in train)"))
e = first(kk[kk.exact & ~same_pid])
if e:
    examples.append((e, "IDENTICAL image filed under two\npatient numbers (both in test)"))
e = first(kk[~kk.exact & (kk.category == "kermany train+val vs kermany test")].sort_values("c256", ascending=False))
if e:
    examples.append((e, f"Closest train–test pair: correlation {e[2]:.2f}.\nDifferent children, NOT a copy"))
rr_ = pairs[pairs.kind == 2]
e = first(rr_[rr_.exact])
if e:
    examples.append((e, "IDENTICAL image under two image ids\n(adult dataset, both in test)"))
cross = rr_[~rr_.exact & (rr_.category == "rsna adapt vs rsna test")]
e = first((cross[cross.same_nih_patient] if nihv is not None else cross).sort_values("c256", ascending=False))
if e:
    examples.append((e, f"Same adult, two visits, one in 'adapt' and\none in 'test' (correlation {e[2]:.2f})"))
e = first(pairs[pairs.kind == 1].sort_values("c256", ascending=False))
if e:
    examples.append((e, f"Closest child–adult pair: correlation {e[2]:.2f}.\nNo image is in both datasets"))


def caption(g):
    if g < nk:
        row = mk.iloc[g]
        return f"{row.filename.replace('.jpeg', '')}\n{row.split}"
    row = mr.iloc[g - nk]
    return f"{row.age} y, {row.sex}, {row['view']}, {row.cls.split(' /')[0].lower()}\n{row.split}"


ncol = 2
nrow = int(np.ceil(len(examples) / ncol))
fig, axes = plt.subplots(nrow, ncol * 2, figsize=(12.5, 3.75 * nrow), gridspec_kw=dict(wspace=0.08, hspace=0.55))
axes = np.atleast_2d(axes)
for ax in axes.ravel():
    ax.axis("off")
for kx, ((i, j, c), title) in enumerate(examples):
    for sx, g in enumerate((i, j)):
        ax = axes[kx // ncol, (kx % ncol) * 2 + sx]
        ax.axis("on")
        ax.imshow(image(g), cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(4)
            sp.set_color(plotting.ADULTS if g >= nk else plotting.CHILDREN)
        ax.set_xlabel(caption(g), fontsize=9.5, color=plotting.TEXT_2)
        if sx == 0:
            ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold")
fig.legend(handles=[Patch(color=plotting.CHILDREN, label="Children's dataset (Kermany)"),
                    Patch(color=plotting.ADULTS, label="Adult dataset (RSNA)")],
           loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.02))
k_id = identical["kermany"]
fig.suptitle(f"No X-ray is shared between training and test data. {k_id['redundant_images']} children's images are stored twice "
             f"inside one split,\nand in the adult dataset the same patients appear on both sides of the split",
             x=0.04, ha="left", fontsize=14, fontweight="bold", y=0.99)
plotting.finish(fig, C.FIGURES_DIR / "audit_duplicate_pairs.png",
                note=f"All {n * (n - 1) // 2:,} image pairs compared. Same image = pixel correlation of at least {NEAR_DUP_CORR_256} at 256 px.")

# ==========================================================================
# A3 / A4. Shortcuts: predicting the label without looking at the lungs
# ==========================================================================
LR_GRID = [0.01, 0.1, 1.0, 10.0]


def shortcut(F_train, y_train, F_val, y_val, evals, feature_names):
    """Fit logistic regression and gradient-boosted trees on train, keep the one with the better val AUROC.
    evals: {name: (features, labels)} test groups that are only scored."""
    F_train, F_val = np.asarray(F_train, np.float64), np.asarray(F_val, np.float64)
    sc = StandardScaler().fit(F_train)
    best = None
    for c in LR_GRID:
        clf = LogisticRegression(C=c, max_iter=5000).fit(sc.transform(F_train), y_train)
        v = metrics.auroc(y_val, clf.predict_proba(sc.transform(F_val))[:, 1])
        if best is None or v > best[0]:
            best = (v, c, clf)
    lr_val, lr_c, lr = best
    gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=8, l2_regularization=1.0,
                                        early_stopping=False, random_state=C.SEED).fit(F_train, y_train)
    gb_val = metrics.auroc(y_val, gb.predict_proba(F_val)[:, 1])
    models = {"logistic regression": (lr_val, lambda F: lr.predict_proba(sc.transform(np.asarray(F, np.float64)))[:, 1]),
              "gradient-boosted trees": (gb_val, lambda F: gb.predict_proba(np.asarray(F, np.float64))[:, 1])}
    chosen = max(models, key=lambda k: models[k][0])
    out = dict(features=feature_names, n_features=int(F_train.shape[1]), n_train=int(len(y_train)),
               model_chosen_on_validation=chosen, validation_auroc={k: v[0] for k, v in models.items()},
               logistic_regression_C=lr_c)
    for name, (F, y) in evals.items():
        out[name] = auroc_ci(y, models[chosen][1](F))
        out[name]["auroc_of_the_other_model"] = metrics.auroc(y, models[[k for k in models if k != chosen][0]][1](F))
    return out


log("A3 shortcuts in the children's dataset")
size = np.c_[mk.orig_w.values, mk.orig_h.values, mk.orig_w.values / mk.orig_h.values]
pct_names = ["mean", "std"] + [f"p{p}" for p in PERCENTILES]
adult_evals = {KEY_ADULT_NORMAL: r_test & vs_normal, KEY_ADULT_ALL: r_test}
a3 = {}
for name, Fk_, Fr_, names, what in [
        ("image_size", size, None, ["orig_w", "orig_h", "orig_w / orig_h"],
         "Width, height and aspect ratio of the original JPEG file. No pixel is used."),
        ("brightness", fk["bright"], fr["bright"], pct_names,
         "Mean, standard deviation and 7 percentiles of the grey values of the 224 px centre crop. No shapes, no positions."),
        ("border_cut_off_at_test_time", fk["border_outer"], fr["border_outer"], ["60 block means"],
         "Outer 16-pixel frame of the stored 256 px image (60 blocks of 16x16 px). The network never sees this frame at "
         "test time (224 px centre crop), only partly during training augmentation."),
        ("border_the_network_sees", fk["border_seen"], fr["border_seen"], ["52 block means"],
         "Outer 16-pixel frame of the 224 px centre crop, i.e. the rim of what the network sees (52 blocks): "
         "shoulders, arms, neck, belly, letters and collimator edges, hardly any lung.")]:
    evals = {KEY_CHILD: (Fk_[k_test], yk[k_test])}
    if Fr_ is not None:   # does the children's shortcut also "work" on adults? (trained on children only)
        evals.update({k: (Fr_[m], yr[m]) for k, m in adult_evals.items()})
    a3[name] = dict(what=what, **shortcut(Fk_[k_train], yk[k_train], Fk_[k_val], yk[k_val], evals, names))
    log(f"   {name:30s} children test AUROC {a3[name][KEY_CHILD]['auroc']:.3f}  ({a3[name]['model_chosen_on_validation']})")
audit["A3_shortcuts_children"] = dict(
    protocol="Fitted on Kermany train, model type and C chosen on Kermany val, AUROC on Kermany test. "
             "For brightness and border the same children-trained model is also scored on RSNA test adults.",
    chance_level=0.5, **a3)

log("A4 shortcuts in the adult dataset")
meta = np.c_[(mr["view"] == "AP").values, mr.age.values, (mr.sex == "M").values].astype(float)
a4 = {}
for name, colsx, names in [("view_only", [0], ["view is AP"]), ("age_only", [1], ["age"]), ("sex_only", [2], ["sex is male"]),
                           ("view_age_sex", [0, 1, 2], ["view is AP", "age", "sex is male"])]:
    res = {}
    for key, keep in ((KEY_ADULT_NORMAL, vs_normal), (KEY_ADULT_ALL, np.ones(nr, bool))):
        tr, va, te = r_adapt & keep, r_val & keep, r_test & keep
        s = shortcut(meta[tr][:, colsx], yr[tr], meta[va][:, colsx], yr[va], {"test": (meta[te][:, colsx], yr[te])}, names)
        res[key] = dict(model_chosen_on_validation=s["model_chosen_on_validation"], n_train=s["n_train"], **s["test"])
    a4[name] = dict(features=names, **res)
    log(f"   {name:14s} AUROC {res[KEY_ADULT_NORMAL]['auroc']:.3f} (vs normal)  {res[KEY_ADULT_ALL]['auroc']:.3f} (vs everything else)")
audit["A4_shortcuts_adults"] = dict(
    protocol="Fitted on RSNA adapt adults, model type chosen on RSNA val adults, AUROC on RSNA test adults. No pixel is used.",
    chance_level=0.5, **a4,
    lung_opacity_rate_by_view_in_adult_test_set=r_tables["adult_test_set"]["by_view"])

# Figure: all shortcuts in one picture
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9), sharex=True, gridspec_kw=dict(wspace=0.75))
left = [("image_size", "Size of the image file only\n(width, height)"), ("brightness", "Overall brightness only\n(9 numbers)"),
        ("border_the_network_sees", "Rim of the picture only\n(what the network sees)"),
        ("border_cut_off_at_test_time", "Outermost frame only\n(cropped away at test time)")]
right = [("view_only", "View only\n(bedside AP or standing PA)"), ("age_only", "Age only"), ("sex_only", "Sex only"),
         ("view_age_sex", "View + age + sex")]
for ax, rows_, src, key, colour, title in (
        (axes[0], left, a3, KEY_CHILD, plotting.CHILDREN, "Children: pneumonia vs normal"),
        (axes[1], right, a4, KEY_ADULT_ALL, plotting.ADULTS, "Adults: lung opacity vs everything else")):
    for i, (k, _) in enumerate(rows_):
        v = src[k][key]
        ax.plot(v["auroc_ci"], [i, i], color=colour, lw=2.5, solid_capstyle="round", zorder=2)
        ax.plot(v["auroc"], i, "o", ms=11, color=colour, zorder=3)
        ax.annotate(f"{v['auroc']:.2f}", (max(v["auroc_ci"][1], v["auroc"]), i), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=12, color=plotting.TEXT)
    ax.axvline(0.5, color=plotting.NEUTRAL, lw=1.5, zorder=1)
    ax.annotate("coin flip", (0.5, len(rows_) - 0.42), xytext=(5, 0), textcoords="offset points", fontsize=10.5,
                color=plotting.TEXT_2, va="center")
    ax.set_yticks(range(len(rows_)), [lab for _, lab in rows_])
    ax.set_ylim(len(rows_) - 0.25, -0.6)
    ax.set_xlim(0.38, 1.04)
    ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0, labelsize=11.5)
    ax.set_title(title, fontsize=12.5, fontweight="normal", color=plotting.TEXT_2)
    ax.set_xlabel("AUROC of a model that never looks at the lungs")
fig.suptitle(f"The label can be guessed without looking at the lungs: from the image size alone in children "
             f"(AUROC {a3['image_size'][KEY_CHILD]['auroc']:.2f}),\nfrom the type of X-ray alone in adults "
             f"(AUROC {a4['view_only'][KEY_ADULT_ALL]['auroc']:.2f})", x=0.01, ha="left", fontsize=14, fontweight="bold", y=1.04)
plotting.finish(fig, C.FIGURES_DIR / "audit_shortcuts.png",
                note=f"Children: fitted on Kermany train, scored on Kermany test (n={k_test.sum()}). Adults: fitted on RSNA adapt, scored on RSNA "
                     f"test adults (n={r_test.sum():,}).\nLines: 95% bootstrap intervals. AUROC 0.5 = coin flip, 1.0 = perfect.")

# Figure: the image-size shortcut itself
fig, axes = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True, gridspec_kw=dict(hspace=0.32))
bins = np.arange(300, 2800, 50)
for ax, lab, name in ((axes[0], 0, "Normal"), (axes[1], 1, "Pneumonia")):
    h = mk.orig_h.values[k_train & (yk == lab)]
    ax.hist(h, bins=bins, color=plotting.CHILDREN, rwidth=0.88, zorder=3)
    ax.axvline(np.median(h), color=plotting.TEXT, lw=1.5, zorder=4)
    ax.annotate(f"median {np.median(h):,.0f} px", (np.median(h), 1), xycoords=("data", "axes fraction"), xytext=(6, -4),
                textcoords="offset points", va="top", fontsize=11, color=plotting.TEXT)
    ax.set_title(f"{name} ({len(h):,} training images)", fontsize=12, fontweight="normal", color=plotting.TEXT_2)
    ax.set_ylabel("Images")
    ax.grid(axis="x", visible=False)
axes[1].set_xlabel("Height of the original image file (pixels)")
fig.suptitle(f"Healthy children's X-rays are simply bigger files: width and height alone separate\nnormal from pneumonia "
             f"with AUROC {a3['image_size'][KEY_CHILD]['auroc']:.2f} on the test set, without a single pixel",
             x=0.01, ha="left", fontsize=14, fontweight="bold", y=1.04)
plotting.finish(fig, C.FIGURES_DIR / "audit_image_size_shortcut.png",
                note="Kermany train. All images are resized to 256 px before training, so the network cannot read the file size directly;\n"
                     "the gap shows that normal and pneumonia images were produced differently (cropping, equipment or export settings).")

# ==========================================================================
# A5. "Which hospital is this from?"
# ==========================================================================
log("A5 which hospital")
HOSP_C = 0.01          # fixed, not tuned
rng = np.random.default_rng(C.SEED)
Tk, Tr = fk["thumb224"], fr["thumb224"]
young5 = (mr.age_valid & (mr.age <= 5)).values
under18 = (mr.age_valid & (mr.age < 18)).values
r_groups_id = nihv if nihv is not None else mr.patient_id.values.astype(str)


def fit_source(Fa, Fb):
    """Logistic regression 'does this picture come from set B?' on standardised features."""
    F = np.concatenate([Fa, Fb]).astype(np.float64)
    y = np.r_[np.zeros(len(Fa)), np.ones(len(Fb))]
    sc = StandardScaler().fit(F)
    clf = LogisticRegression(C=HOSP_C, class_weight="balanced", max_iter=5000).fit(sc.transform(F), y)
    return lambda X: clf.predict_proba(sc.transform(np.asarray(X, np.float64)))[:, 1]


def score_source(pa, pb):
    out = auroc_ci(np.r_[np.zeros(len(pa)), np.ones(len(pb))], np.r_[pa, pb])
    out.pop("n_pos")
    out.update(n_a=int(len(pa)), n_b=int(len(pb)), share_of_a_called_a=float((pa < 0.5).mean()), share_of_b_called_b=float((pb >= 0.5).mean()))
    return out


def cross_validated(Fa, ga, Fb, gb_, folds=5):
    """Out-of-fold scores; folds keep every patient entirely on one side."""
    F = np.concatenate([Fa, Fb])
    y = np.r_[np.zeros(len(Fa)), np.ones(len(Fb))].astype(int)
    groups = np.r_[["a_" + str(g) for g in ga], ["b_" + str(g) for g in gb_]]
    oof = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(folds, shuffle=True, random_state=C.SEED).split(F, y, groups):
        oof[te] = fit_source(F[tr][y[tr] == 0], F[tr][y[tr] == 1])(F[te])
    return score_source(oof[y == 0], oof[y == 1])


a5 = {}
# (1) children (Kermany) vs adults (RSNA): age AND hospital differ. Train and test on disjoint images.
adapt_pick = np.sort(rng.choice(np.flatnonzero(r_adapt), k_train.sum(), replace=False))
src = fit_source(Tk[k_train], Tr[adapt_pick])
a5["children_vs_adults"] = dict(
    question="Kermany children vs RSNA adults: both age and hospital differ",
    trained_on=f"Kermany train ({k_train.sum()}) vs a random {len(adapt_pick)} RSNA adapt adults",
    tested_on="Kermany test vs all RSNA test adults", **score_source(src(Tk[k_test]), src(Tr[r_test])))
src_b = fit_source(fk["bright"][k_train], fr["bright"][adapt_pick])
a5["children_vs_adults_from_brightness_only"] = dict(
    question="the same, but from 9 brightness numbers instead of the picture",
    **score_source(src_b(fk["bright"][k_test]), src_b(fr["bright"][r_test])))
# (2) the same classifier, shown RSNA children it has never seen: does it still say "RSNA"?
for name, m in (("age_1_to_5", young5), ("under_18", under18)):
    a5[f"adult_trained_detector_on_rsna_children_{name}"] = dict(
        question=f"classifier from (1), never shown an RSNA child: Kermany test vs RSNA children {name.replace('_', ' ')} (all splits)",
        **score_source(src(Tk[k_test]), src(Tr[m])))
# (3) same age, different hospital: cross-validated, Kermany train+val vs RSNA children
k_dev = k_train | k_val
for name, m in (("age_1_to_5", young5), ("under_18", under18)):
    a5[f"same_age_different_hospital_{name}"] = dict(
        question=f"Kermany children (train+val, test set untouched) vs RSNA children {name.replace('_', ' ')}: only the hospital differs"
                 + ("" if name == "age_1_to_5" else " (ages overlap only partly)"),
        how="5-fold cross-validation, every patient entirely inside one fold",
        **cross_validated(Tk[k_dev], mk.patient_id.values[k_dev], Tr[m], r_groups_id[m]))
a5["same_age_different_hospital_age_1_to_5_from_brightness_only"] = dict(
    question="the same-age comparison from 9 brightness numbers only", how="5-fold cross-validation",
    **cross_validated(fk["bright"][k_dev], mk.patient_id.values[k_dev], fr["bright"][young5], r_groups_id[young5]))
# (4) same hospital, different age: RSNA adults vs RSNA children aged 1-5
adult_pick = np.sort(rng.choice(np.flatnonzero(adult), 5000, replace=False))
a5["same_hospital_different_age"] = dict(
    question="RSNA adults (random 5000) vs RSNA children aged 1-5: only the age differs",
    how="5-fold cross-validation, every patient entirely inside one fold",
    **cross_validated(Tr[adult_pick], r_groups_id[adult_pick], Tr[young5], r_groups_id[young5]))
young = mr[young5]
a5["how_small_the_rsna_child_groups_are"] = dict(
    age_1_to_5=dict(n_images=int(young5.sum()), n_patients=int(len(set(r_groups_id[young5]))) if nihv is not None else None,
                    by_class={str(k): int(v) for k, v in young.cls.value_counts().items()},
                    by_view={str(k): int(v) for k, v in young["view"].value_counts().items()},
                    by_split={str(k): int(v) for k, v in young.split.value_counts().items()}),
    under_18=dict(n_images=int(under18.sum()), n_patients=int(len(set(r_groups_id[under18]))) if nihv is not None else None,
                  by_class={str(k): int(v) for k, v in mr[under18].cls.value_counts().items()},
                  by_view={str(k): int(v) for k, v in mr[under18]["view"].value_counts().items()}))
audit["A5_which_hospital"] = dict(
    method=f"Logistic regression (C={HOSP_C}, fixed, not tuned) on the 224 px centre crop shrunk to 32x32 pixels, standardised. "
           "Group a = Kermany (or RSNA adults in the last comparison), group b = RSNA (children). AUROC 0.5 = cannot tell apart.",
    caveat="'Hospital' here bundles everything that differs between the two sources: scanner, export and compression, "
           "positioning, country and patient mix. The RSNA children are few and are not a random sample of children.",
    **a5)
for k, v in a5.items():
    if "auroc" in v:
        log(f"   {k:60s} AUROC {v['auroc']:.3f}  (n {v['n_a']} vs {v['n_b']})")

# Figure: same age, different hospital
same = a5["same_age_different_hospital_age_1_to_5"]
show_k = np.sort(rng.choice(np.flatnonzero(k_test), 6, replace=False))
show_r = np.sort(rng.choice(np.flatnonzero(young5), 6, replace=False))
fig, axes = plt.subplots(2, 6, figsize=(12.5, 5.2), gridspec_kw=dict(wspace=0.06, hspace=0.22))
for row, (idxs, X, colour) in enumerate(((show_k, Xk, plotting.CHILDREN), (show_r, Xr, plotting.ADULTS))):
    for ax, g in zip(axes[row], idxs):
        ax.imshow(center_crop(np.asarray(X[g]), C.IMG_SIZE), cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(4)
            sp.set_color(colour)
        if row == 1:
            ax.set_xlabel(f"{mr.age.iloc[g]} years, {mr['view'].iloc[g]}", fontsize=10.5)
axes[0, 0].set_title("Guangzhou (Kermany), ages 1–5", loc="left", fontsize=12, fontweight="normal", color=plotting.TEXT_2)
axes[1, 0].set_title(f"USA (RSNA), ages 1–5: only {same['n_b']} X-rays exist", loc="left", fontsize=12, fontweight="normal",
                     color=plotting.TEXT_2)
fig.suptitle(f"Same age, different hospital: a simple classifier still tells the two sources apart (AUROC {same['auroc']:.2f}).\n"
             "So the model faces a new hospital, not only a new age group", x=0.02, ha="left", fontsize=14, fontweight="bold", y=1.06)
plotting.finish(fig, C.FIGURES_DIR / "audit_which_hospital.png",
                note=f"Random examples as the network sees them (224 px centre crop). Logistic regression on 32×32 pixels, 5-fold cross-validation, "
                     f"{same['n_a']:,} Kermany vs {same['n_b']} RSNA X-rays.\nWith only {same['n_b']} US children the estimate is rough "
                     f"(95% interval {same['auroc_ci'][0]:.2f}–{same['auroc_ci'][1]:.2f}), and they are not a random sample of children.")

# ==========================================================================
(C.RESULTS_DIR / "audit.json").write_text(json.dumps(clean(audit), indent=2))
log(f"wrote {C.RESULTS_DIR / 'audit.json'} and figures audit_*.png")
