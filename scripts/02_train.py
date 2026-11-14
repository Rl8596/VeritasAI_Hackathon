"""Step 2: fine-tune a model and export its predictions.

    python scripts/02_train.py --name kids_only                      # the "tutorial" model: children only
    python scripts/02_train.py --name kids_plus_adults --rsna-n 6000  # children + 6000 adult X-rays

Outputs go to artifacts/<name>/ (weights, config, predictions for every image)
and results/headline_<name>.json (children vs adults, same threshold).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pneumoshift import config as C, data, train  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
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
    args = ap.parse_args()

    Xk, mk = data.load_kermany()
    Xr, mr = data.load_rsna()
    out_dir = C.ARTIFACTS_DIR / args.name
    rng = np.random.default_rng(args.seed)

    train_sets, val_parts, desc = [], [], []
    if not args.no_kids:
        m = (mk.split == "train").values
        train_sets.append((Xk, mk.idx[m].values, mk.label[m].values))
        desc.append(f"{m.sum()} children X-rays (Kermany train)")
    if args.rsna_n:
        pool = mr.idx[(mr.split == "adapt") & mr.is_adult].values
        pick = np.sort(rng.choice(pool, size=min(args.rsna_n, len(pool)), replace=False))
        train_sets.append((Xr, pick, mr.label.values[pick]))
        desc.append(f"{len(pick)} adult X-rays (RSNA adapt)")

    # Validation: children val, plus adult val when adults are part of training.
    # It is used for model selection, threshold and temperature. Test data is never used here.
    if not args.no_kids:
        m = (mk.split == "val").values
        val_k = (Xk, mk.idx[m].values, mk.label[m].values)
    if args.rsna_n:
        m = ((mr.split == "val") & mr.is_adult).values
        val_r = (Xr, mr.idx[m].values, mr.label[m].values)

    if args.rsna_n and not args.no_kids:
        # one combined validation array so that the threshold has to work for both groups
        Xv = np.concatenate([np.asarray(Xk[val_k[1]]), np.asarray(Xr[val_r[1]])])
        val_set = (Xv, np.arange(len(Xv)), np.concatenate([val_k[2], val_r[2]]))
        val_desc = "children val + adult val"
    elif args.rsna_n:
        val_set, val_desc = val_r, "adult val"
    else:
        val_set, val_desc = val_k, "children val"

    print(f"Training '{args.name}' on " + " + ".join(desc) + f"; validation: {val_desc}")
    model, cfg = train.train_model(
        train_sets, val_set, out_dir, arch=args.arch, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        threads=args.threads, workers=args.workers, seed=args.seed,
        extra_config=dict(name=args.name, train_data=desc, val_data=val_desc))
    if args.rsna_n:
        np.save(out_dir / "rsna_train_idx.npy", train_sets[-1][1])

    train.export_predictions(model, out_dir, {"kermany": Xk, "rsna": Xr})
    head = train.headline(out_dir, mk, mr)
    C.RESULTS_DIR.mkdir(exist_ok=True)
    (C.RESULTS_DIR / f"headline_{args.name}.json").write_text(json.dumps(
        dict(model=args.name, threshold=cfg["threshold"], temperature=cfg["temperature"], results=head), indent=2))
    print(f"\nthreshold {cfg['threshold']:.3f} (95% sensitivity on validation), temperature {cfg['temperature']:.2f}")
    for k, v in head.items():
        print(f"{k:55s} n={v['n']:5d}  AUROC {v['auroc']:.3f}  sens {v['sensitivity']:.3f}  spec {v['specificity']:.3f}  "
              f"acc {v['accuracy']:.3f}  (majority baseline {v['majority_baseline_accuracy']:.3f})")
