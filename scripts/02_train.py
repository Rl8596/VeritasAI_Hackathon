"""Step 2: fine-tune a model and export its predictions.

    python scripts/02_train.py --name kids_only                      # the "tutorial" model: children only
    python scripts/02_train.py --name kids_plus_adults --rsna-n 6000  # children + 6000 adult X-rays

Outputs go to artifacts/<name>/ (weights, config, predictions for every image),
results/val_<name>.json (validation results) and, unless --no-test is given,
results/headline_<name>.json (test results: children vs adults, same threshold).

A name that already has a trained model is refused, so that no model is overwritten by accident.

Options added for training on a GPU (scripts/09_gpu_batch.py uses them):
    --val adults    choose the best epoch, the temperature and the threshold on ADULT validation data only
                    (default "auto": children + adults pooled when both are in training, as for the first models)
    --no-test       do not compute or show test results. Use it whenever several candidate models are trained:
                    candidates are compared on validation data, test results are computed once at the end.
    --select last   keep the last epoch instead of the best one on validation data
    --amp           mixed precision (faster, half the GPU memory)
    --require-gpu   stop instead of silently training on the CPU
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pneumoshift import config as C, data, train  # noqa: E402

# The published first models. The live demo loads them, they must never be replaced.
PROTECTED = ("kids_only", "kids_plus_adults")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--rsna-n", type=int, default=0, help="number of adult 'adapt' images to add to training")
    ap.add_argument("--no-kids", action="store_true", help="train on adults only")
    ap.add_argument("--arch", default=C.DEFAULT_ARCH)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--val", choices=["auto", "pooled", "adults", "children"], default="auto",
                    help="validation data for best epoch, temperature and threshold")
    ap.add_argument("--select", choices=["best", "last"], default="best", help="which epoch is kept")
    ap.add_argument("--no-test", action="store_true", help="do not compute or print test results")
    ap.add_argument("--reproduce-published", action="store_true",
                    help="allow the names kids_only and kids_plus_adults (the Colab notebook retrains the published models)")
    ap.add_argument("--amp", action="store_true", help="mixed precision on the GPU")
    ap.add_argument("--require-gpu", action="store_true", help="stop if no GPU is found")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing model of the same name")
    args = ap.parse_args()

    out_dir = C.ARTIFACTS_DIR / args.name
    if args.name in PROTECTED and not args.reproduce_published:
        sys.exit(f"'{args.name}' is the name of a published first model. New models get new names. "
                 "(To retrain the published recipe as a reproduction, add --reproduce-published.)")
    if (out_dir / "model.safetensors").exists():
        if args.name in PROTECTED and os.environ.get("PNEUMOSHIFT_ALLOW_PROTECTED") != "1":
            sys.exit(f"{out_dir} holds a published first model. It is not replaced.")
        if not args.overwrite:
            sys.exit(f"{out_dir} already holds a trained model. Choose another --name, or pass --overwrite to replace it.")
    if args.no_kids and not args.rsna_n:
        sys.exit("--no-kids needs --rsna-n: there would be nothing to train on")

    Xk, mk = data.load_kermany()
    Xr, mr = data.load_rsna()
    rng = np.random.default_rng(args.seed)

    train_sets, desc, groups = [], [], []
    if not args.no_kids:
        m = (mk.split == "train").values
        train_sets.append((Xk, mk.idx[m].values, mk.label[m].values))
        desc.append(f"{m.sum()} children X-rays (Kermany train)")
        groups.append("children")
    if args.rsna_n:
        pool = mr.idx[(mr.split == "adapt") & mr.is_adult].values
        pick = np.sort(rng.choice(pool, size=min(args.rsna_n, len(pool)), replace=False))
        train_sets.append((Xr, pick, mr.label.values[pick]))
        desc.append(f"{len(pick)} adult X-rays (RSNA adapt)")
        groups.append("adults")

    # Validation data. It is used for model selection, threshold and temperature. Test data is never used here.
    m = (mk.split == "val").values
    val_k = (Xk, mk.idx[m].values, mk.label[m].values)
    adult_val = ((mr.split == "val") & mr.is_adult).values
    val_r = (Xr, mr.idx[adult_val].values, mr.label[adult_val].values)

    choice = args.val
    if choice == "auto":
        choice = "pooled" if (args.rsna_n and not args.no_kids) else "adults" if args.rsna_n else "children"
    if choice == "pooled":
        # one combined validation array so that the threshold has to work for both groups
        Xv = np.concatenate([np.asarray(Xk[val_k[1]]), np.asarray(Xr[val_r[1]])])
        val_set = (Xv, np.arange(len(Xv)), np.concatenate([val_k[2], val_r[2]]))
        val_desc = "children val + adult val"
    elif choice == "adults":
        val_set, val_desc = val_r, "adult val"
    else:
        val_set, val_desc = val_k, "children val"

    # Reported for every model, whatever it was trained on: validation results per group and per X-ray view.
    report_sets = {"children": val_k, "adults": val_r}
    for view in ("PA", "AP"):
        v = adult_val & (mr["view"] == view).values
        report_sets[f"adults_{view}"] = (Xr, mr.idx[v].values, mr.label[v].values)

    print(f"Training '{args.name}' on " + " + ".join(desc) + f"; validation: {val_desc}", flush=True)
    model, cfg = train.train_model(
        train_sets, val_set, out_dir, arch=args.arch, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        threads=args.threads, workers=args.workers, seed=args.seed, amp=args.amp, require_gpu=args.require_gpu,
        report_sets=report_sets, select=args.select,
        extra_config=dict(name=args.name, train_data=desc, train_groups=groups, val_data=val_desc))
    if args.rsna_n:
        np.save(out_dir / "rsna_train_idx.npy", train_sets[-1][1])

    train.export_predictions(model, out_dir, {"kermany": Xk, "rsna": Xr})
    C.RESULTS_DIR.mkdir(exist_ok=True)
    (C.RESULTS_DIR / f"val_{args.name}.json").write_text(json.dumps(dict(
        model=args.name, arch=cfg["arch"], train_data=desc, val_data=val_desc, threshold=cfg["threshold"],
        temperature=cfg["temperature"], best_epoch=cfg["best_epoch"], epoch_rule=cfg["epoch_rule"], epochs=cfg["epochs"], seed=cfg["seed"],
        train_minutes=cfg["train_minutes"], device=cfg["device"], mixed_precision=cfg["mixed_precision"],
        note="Validation data only. No test image was used for these numbers.",
        validation=cfg["validation"], history=cfg["history"]), indent=2))
    print(f"\nthreshold {cfg['threshold']:.3f} (95% sensitivity on {val_desc}), temperature {cfg['temperature']:.2f}, "
          f"kept epoch {cfg['best_epoch']} of {cfg['epochs']} (rule: {cfg['epoch_rule']})")
    print("VALIDATION results (no test image involved):")
    for k, v in cfg["validation"].items():
        print(f"  {k:12s} n={v['n']:5d}  AUROC {v['auroc']:.3f}  sens {v['sensitivity']:.3f}  spec {v['specificity']:.3f}")

    if args.no_test:
        print("\nTest results were not computed (--no-test). They come from scripts/05_shift_analysis.py at the end.")
    else:
        head = train.headline(out_dir, mk, mr)
        (C.RESULTS_DIR / f"headline_{args.name}.json").write_text(json.dumps(
            dict(model=args.name, threshold=cfg["threshold"], temperature=cfg["temperature"], results=head), indent=2))
        print("\nTEST results:")
        for k, v in head.items():
            print(f"{k:55s} n={v['n']:5d}  AUROC {v['auroc']:.3f}  sens {v['sensitivity']:.3f}  spec {v['specificity']:.3f}  "
                  f"acc {v['accuracy']:.3f}  (majority baseline {v['majority_baseline_accuracy']:.3f})")
