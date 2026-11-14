"""Step 6: does the model look at the right place? And what does it get wrong?

    python scripts/06_heatmap_check.py --models kids_only kids_plus_adults

Part 1, the pointing test. Radiologists drew boxes around every lung opacity in the
adult dataset. For each adult test X-ray with pneumonia we take the single hottest
point of the model's Grad-CAM heatmap and check whether it lies inside a radiologist's
box. Three reference points show what "no skill" looks like: a random point, the image
centre, and the one fixed spot where opacities are most common (learned on other images).

Part 2, the failure gallery: the most confident mistakes of each model, with heatmaps.

Writes results/heatmap_check.json and figures results/figures/heatmap_*.png.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pneumoshift import cam, config as C, data, modeling, plotting  # noqa: E402
from pneumoshift.preprocess import center_crop, to_model_input  # noqa: E402

SCALE = C.STORE_SIZE / 1024                      # RSNA originals are 1024 px
OFFSET = (C.STORE_SIZE - C.IMG_SIZE) // 2        # centre crop 256 -> 224


def boxes_in_model_coords(boxes_df) -> dict:
    """patient_id -> list of (x0, y0, x1, y1) in the 224 px image the model sees."""
    out = {}
    for r in boxes_df.itertuples():
        x0, y0 = r.x * SCALE - OFFSET, r.y * SCALE - OFFSET
        out.setdefault(r.patient_id, []).append((x0, y0, x0 + r.width * SCALE, y0 + r.height * SCALE))
    return out


def box_mask(bxs) -> np.ndarray:
    m = np.zeros((C.IMG_SIZE, C.IMG_SIZE), dtype=bool)
    for x0, y0, x1, y1 in bxs:
        m[max(0, int(round(y0))):min(C.IMG_SIZE, int(round(y1))), max(0, int(round(x0))):min(C.IMG_SIZE, int(round(x1)))] = True
    return m


def cams_for(model, X, idx, batch=32):
    maps = []
    for i in range(0, len(idx), batch):
        x = to_model_input(np.asarray(X[np.sort(idx[i:i + batch])]))
        order = np.argsort(np.argsort(idx[i:i + batch]))
        maps.append(cam.grad_cam(model, x)[0][order])
    return np.concatenate(maps)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["kids_only"])
    ap.add_argument("--max-images", type=int, default=1500)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plotting.use_style()
    Xk, mk = data.load_kermany()
    Xr, mr = data.load_rsna()
    boxes = boxes_in_model_coords(data.load_rsna_boxes())
    models = {m: modeling.load_checkpoint(C.ARTIFACTS_DIR / m) for m in args.models if (C.ARTIFACTS_DIR / m).exists()}
    res = {}

    # ---------------- Part 1: pointing test on adult test X-rays with pneumonia ----------------
    sel = mr.idx[(mr.split == "test") & mr.is_adult & (mr.label == 1)].values
    rng = np.random.default_rng(C.SEED)
    if len(sel) > args.max_images:
        sel = np.sort(rng.choice(sel, args.max_images, replace=False))
    masks = np.stack([box_mask(boxes[mr.patient_id[i]]) for i in sel])

    # where are opacities most common? learned from the "adapt" images, never from test images
    prior = np.zeros((C.IMG_SIZE, C.IMG_SIZE))
    for i in mr.idx[(mr.split == "adapt") & (mr.label == 1)].values:
        prior += box_mask(boxes[mr.patient_id[i]])
    py, px = np.unravel_index(prior.argmax(), prior.shape)
    c = C.IMG_SIZE // 2
    res["pointing_test"] = dict(
        n_images=int(len(sel)),
        reference=dict(random_point=float(masks.mean()), image_centre=float(masks[:, c, c].mean()),
                       most_common_spot=float(masks[:, py, px].mean())),
        models={})
    all_cams = {}
    for name, (model, cfg) in models.items():
        cams = cams_for(model, Xr, sel)
        all_cams[name] = cams
        flat = cams.reshape(len(cams), -1).argmax(1)
        hy, hx = np.unravel_index(flat, cams.shape[1:])
        hit = masks[np.arange(len(sel)), hy, hx]
        inside = (cams * masks).sum((1, 2)) / (cams.sum((1, 2)) + 1e-8)
        # 95% interval for the hit rate
        boot = [hit[rng.integers(0, len(hit), len(hit))].mean() for _ in range(1000)]
        res["pointing_test"]["models"][name] = dict(
            hit_rate=float(hit.mean()), hit_rate_ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            share_of_heat_inside_boxes=float(inside.mean()))

    # figure: hit rates
    pt = res["pointing_test"]
    rows = [("Random point", pt["reference"]["random_point"], plotting.NEUTRAL),
            ("Image centre", pt["reference"]["image_centre"], plotting.NEUTRAL),
            ("Most common spot", pt["reference"]["most_common_spot"], plotting.NEUTRAL)]
    colours = {"kids_only": plotting.CHILDREN, "kids_plus_adults": plotting.THIRD, "adults_only": plotting.ADULTS}
    nice = {"kids_only": "Children-only model", "kids_plus_adults": "Children + adults model", "adults_only": "Adults-only model"}
    rows += [(nice.get(n, n), v["hit_rate"], colours.get(n, plotting.THIRD)) for n, v in pt["models"].items()]
    fig, ax = plt.subplots(figsize=(7.6, 0.5 * len(rows) + 1.4))
    yy = np.arange(len(rows))[::-1]
    ax.barh(yy, [r[1] for r in rows], color=[r[2] for r in rows], height=0.6)
    for y_, r in zip(yy, rows):
        ax.text(r[1] + 0.01, y_, f"{r[1]:.0%}", va="center", fontsize=11)
    ax.set_yticks(yy, [r[0] for r in rows]); ax.set_xlim(0, 1); ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("Hottest point of the heatmap lies inside a radiologist's box")
    ax.set_title("Does the model look where the radiologists marked pneumonia?")
    plotting.finish(fig, C.FIGURES_DIR / "heatmap_pointing_test.png",
                    note=f"{pt['n_images']} adult test X-rays with lung opacity (RSNA). Grey = no-skill reference points.")

    # figure: the same adult X-rays through each model, radiologist boxes drawn in white
    show = sel[:6]
    fig, axes = plt.subplots(len(models) + 1, len(show), figsize=(2.3 * len(show), 2.3 * (len(models) + 1)), squeeze=False)
    for j, i in enumerate(show):
        gray = center_crop(np.asarray(Xr[i]))
        panels = [("X-ray + radiologist boxes", np.repeat(gray[..., None], 3, 2))]
        panels += [(nice.get(n, n), cam.overlay(gray, all_cams[n][j])) for n in models]
        for r_, (title, img) in enumerate(panels):
            ax = axes[r_, j]
            ax.imshow(img); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for x0, y0, x1, y1 in boxes[mr.patient_id[i]]:
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="white", linewidth=1.5))
            if j == 0:
                ax.set_ylabel(title, fontsize=10, color=plotting.TEXT)
    fig.suptitle("Where each model looks on adult X-rays with pneumonia (white = radiologist)", x=0.01, ha="left",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    plotting.finish(fig, C.FIGURES_DIR / "heatmap_adult_examples.png")

    # ---------------- Part 2: most confident mistakes ----------------
    res["failures"] = {}
    for name, (model, cfg) in models.items():
        T, thr = cfg["temperature"], cfg["threshold"]
        d = C.ARTIFACTS_DIR / name
        pk = modeling.sigmoid(np.load(d / "preds_kermany.npz")["logits"], T)
        pr = modeling.sigmoid(np.load(d / "preds_rsna.npz")["logits"], T)
        groups = [("Children 1-5", Xk, mk, pk, (mk.split == "test").values),
                  ("Adults", Xr, mr, pr, ((mr.split == "test") & mr.is_adult).values)]
        fig, axes = plt.subplots(4, 4, figsize=(9.6, 10.2))
        res["failures"][name] = {}
        row = 0
        for gname, X, meta, p, m in groups:
            y = meta.label.values
            missed = np.flatnonzero(m & (y == 1) & (p < thr))
            false_alarm = np.flatnonzero(m & (y == 0) & (p >= thr))
            res["failures"][name][gname] = dict(n_test=int(m.sum()), missed_pneumonia=int(len(missed)), false_alarms=int(len(false_alarm)))
            for kind, idx, order in (("Missed pneumonia", missed, np.argsort(p[missed])),
                                     ("False alarm", false_alarm, np.argsort(-p[false_alarm]))):
                pick = idx[order][:4]
                cams = cams_for(model, X, pick) if len(pick) else []
                for j in range(4):
                    ax = axes[row, j]; ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
                    if j < len(pick):
                        gray = center_crop(np.asarray(X[pick[j]]))
                        ax.imshow(cam.overlay(gray, cams[j]))
                        extra = f", age {int(meta.age[pick[j]])}, {meta['view'][pick[j]]}" if "age" in meta else ""
                        ax.set_title(f"model says {p[pick[j]]:.0%}{extra}", fontsize=9, fontweight="normal", loc="center")
                    else:
                        ax.axis("off")
                    if j == 0:
                        ax.set_ylabel(f"{gname}\n{kind}", fontsize=10, color=plotting.TEXT)
                row += 1
        fig.suptitle(f"{nice.get(name, name)}: its most confident mistakes", x=0.01, ha="left", fontweight="bold", fontsize=13)
        fig.tight_layout()
        plotting.finish(fig, C.FIGURES_DIR / f"heatmap_failures_{name}.png",
                        note=f"Held-out test images. Flag threshold {thr:.2f}. Heatmap = where the model found evidence for pneumonia.")

    (C.RESULTS_DIR / "heatmap_check.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))
