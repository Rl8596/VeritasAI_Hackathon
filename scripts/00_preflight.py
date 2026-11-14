"""Step 0 on a new training machine: is it safe to train here?

    python scripts/00_preflight.py                # prints PREFLIGHT OK, or says what is wrong and stops
    python scripts/00_preflight.py --fix-split    # if only the train/val/test labels differ: take them from the lock file
    python scripts/00_preflight.py --allow-cpu    # do not insist on a GPU

Run it after scripts/01_prepare_data.py and before any training. It changes nothing (except with --fix-split).

Why it exists: a second machine rebuilds the data and the patient split itself. If its library versions drew
a different split, a model trained there could learn from patients that are test patients elsewhere, and
nobody would notice. So this script checks, in order:

  1. PyTorch sees a GPU (on Windows, a plain "pip install torch" gives a build without GPU support).
  2. Every image row is the same picture, with the same label and the same train/val/test label, as on the
     machine where the first models were made (lock files in data_splits/).
  3. The published children-only model, run on this machine's GPU and this machine's copy of the pictures,
     gives the same scores as the ones stored with the model. That proves the pictures were decoded the
     same way and the GPU computes correctly.
  4. The stored scores of the published second model give its published adult result (AUROC 0.874) with
     this machine's labels and split.
  5. Free disk space, and whether this machine is logged in to Hugging Face (only needed for uploads).

The published models are read from the Hugging Face cache, at the tag v1-before-retraining. Nothing is
written to artifacts/, so this script cannot touch a published model.
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pneumoshift import config as C, data, metrics  # noqa: E402

LOCK = ROOT / "data_splits"
TAG = "v1-before-retraining"
EXPECTED = {"kids_plus_adults adults, lung opacity vs everything else": 0.8739}
problems, warnings = [], []


def say(ok: bool, text: str) -> None:
    print(("  ok    " if ok else "  FAIL  ") + text, flush=True)
    if not ok:
        problems.append(text)


def check_split(name: str, meta: pd.DataFrame, lock: pd.DataFrame, id_cols: list, fix: bool, path: Path) -> pd.DataFrame:
    if len(meta) != len(lock):
        say(False, f"{name}: {len(meta)} rows here, {len(lock)} in the lock file. The data was prepared differently (quick mode?).")
        return meta
    same_rows = all((meta[c].astype(str).values == lock[c].astype(str).values).all() for c in id_cols)
    say(same_rows, f"{name}: every row is the same picture with the same label as in the lock file ({len(meta)} rows)")
    if not same_rows:
        return meta
    differ = int((meta["split"].values != lock["split"].values).sum())
    if differ and fix:
        meta = meta.copy()
        meta["split"] = lock["split"].values
        meta.to_csv(path, index=False)
        print(f"  fixed {name}: {differ} rows now carry the split of the lock file ({path.name} rewritten)")
        differ = 0
    say(differ == 0, f"{name}: train / validation / test labels are identical to the lock file"
        + ("" if differ == 0 else f" ({differ} rows differ; run again with --fix-split)"))
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fix-split", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--offline", action="store_true", help="skip the checks that download the published models")
    args = ap.parse_args()

    print("1. Environment")
    import sklearn
    import timm
    import torch

    print(f"  Python {sys.version.split()[0]}, torch {torch.__version__}, timm {timm.__version__}, numpy {np.__version__}, "
          f"pandas {pd.__version__}, scikit-learn {sklearn.__version__}")
    cuda = torch.cuda.is_available()
    if cuda:
        free, total = torch.cuda.mem_get_info()
        say(True, f"GPU: {torch.cuda.get_device_name(0)}, {free / 2**30:.1f} of {total / 2**30:.1f} GB free")
        if free < 5 * 2**30:
            warnings.append(f"only {free / 2**30:.1f} GB of GPU memory are free: close other programs that use the GPU")
    elif args.allow_cpu:
        warnings.append("no GPU: training will take 20 to 60 minutes per model")
    else:
        say(False, "PyTorch does not see a GPU. Install torch and torchvision from the PyTorch CUDA wheel index.")
    device = "cuda" if cuda else "cpu"

    print("2. Data and split")
    proc = C.PROCESSED_DIR
    if not (proc / "DONE").exists():
        say(False, f"{proc} is not complete. Run: python scripts/01_prepare_data.py")
        sys.exit("\nPREFLIGHT FAILED")
    Xk, mk = data.load_kermany()
    Xr, mr = data.load_rsna()
    mk = check_split("children", mk, pd.read_csv(LOCK / "kermany_split.csv.gz"), ["idx", "filename", "official_split", "label"],
                     args.fix_split, proc / "kermany_meta.csv")
    mr = check_split("adults", mr, pd.read_csv(LOCK / "rsna_split.csv.gz"), ["idx", "patient_id", "label", "is_adult", "view"],
                     args.fix_split, proc / "rsna_meta.csv")
    say(Xk.shape == (len(mk), C.STORE_SIZE, C.STORE_SIZE) and Xr.shape == (len(mr), C.STORE_SIZE, C.STORE_SIZE),
        f"image arrays: children {Xk.shape}, adults {Xr.shape}")
    adults = mr.is_adult.values
    counts = {s: int(((mr.split == s).values & adults).sum()) for s in ("adapt", "val", "test")}
    say(counts == dict(adapt=15142, val=2567, test=7694), f"adult X-rays per split: {counts}")

    if not args.offline and not problems:
        from huggingface_hub import hf_hub_download

        from pneumoshift import modeling

        print("3. Same pictures, same computation: the published children-only model on this machine")
        try:
            repo = "RL8596/pneumoshift-kids-only"
            folder = Path(hf_hub_download(repo, "config.json", revision=TAG)).parent
            hf_hub_download(repo, "model.safetensors", revision=TAG)
            stored_k = np.load(hf_hub_download(repo, "analysis/preds_kermany.npz", revision=TAG))["logits"]
            stored_r = np.load(hf_hub_download(repo, "analysis/preds_rsna.npz", revision=TAG))["logits"]
            model, _ = modeling.load_checkpoint(folder, device)
            rng = np.random.default_rng(0)
            for label, X, stored in (("children", Xk, stored_k), ("adults", Xr, stored_r)):
                idx = np.sort(rng.choice(len(X), size=300, replace=False))
                now = modeling.predict(model, X, idx, device=device)
                d = np.abs(now - stored[idx])
                diff = float(d.max())
                say(diff < 0.1, f"{label}: scores of 300 random X-rays differ from the stored ones by {float(d.mean()):.4f} on average, "
                    f"{diff:.4f} at most (logit units; must stay below 0.1)")
            del model
        except Exception as e:      # no network, or the Hub is down: report, do not pretend
            say(False, f"could not run the published model: {type(e).__name__}: {str(e).splitlines()[0][:160]}")

        print("4. Same labels and split: the published result of the second model, recomputed here")
        try:
            z = np.load(hf_hub_download("RL8596/pneumoshift-kids-plus-adults", "analysis/preds_rsna.npz", revision=TAG))["logits"]
            test = ((mr.split == "test") & mr.is_adult).values
            got = metrics.auroc(mr.label[test], z[test])
            want = EXPECTED["kids_plus_adults adults, lung opacity vs everything else"]
            say(abs(got - want) < 0.0005, f"adult AUROC {got:.4f} (published {want:.4f})")
            used = np.load(hf_hub_download("RL8596/pneumoshift-kids-plus-adults", "analysis/rsna_train_idx.npy", revision=TAG))
            inside = bool(((mr.split.values[used] == "adapt") & mr.is_adult.values[used]).all())
            say(inside, "all 6,000 adult X-rays that model was trained on are training X-rays here (none is validation or test)")
        except Exception as e:
            say(False, f"could not check the published result: {type(e).__name__}: {str(e).splitlines()[0][:160]}")

    print("5. Disk and Hugging Face")
    free_gb = shutil.disk_usage(ROOT).free / 2**30
    say(free_gb > 5, f"{free_gb:.0f} GB free on the disk of the project")
    try:
        from huggingface_hub import HfApi

        who = HfApi().whoami()
        role = ((who.get("auth") or {}).get("accessToken") or {}).get("role")
        print(f"  logged in to Hugging Face as {who.get('name')} (token role: {role})")
        if who.get("name") != "RL8596" or role != "write":
            warnings.append("uploading models needs a WRITE token of the account RL8596 (hf auth login)")
    except Exception:
        print("  not logged in to Hugging Face (training works without it; uploading models does not)")
        warnings.append("not logged in to Hugging Face: models cannot be uploaded from this machine yet")

    for w in warnings:
        print("  note: " + w)
    if problems:
        sys.exit("\nPREFLIGHT FAILED: " + "; ".join(problems))
    print("\nPREFLIGHT OK")
