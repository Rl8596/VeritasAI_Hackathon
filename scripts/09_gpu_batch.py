"""Step 9: train the second generation of models ("v2") on a GPU, one after the other, unattended.

    python scripts/09_gpu_batch.py list                 # the runs, in order, and which are done
    python scripts/09_gpu_batch.py smoke                # 2-minute test of the whole chain, leaves nothing behind
    python scripts/09_gpu_batch.py run --stage 1        # train one stage (1 = must, 2 = should, 3 = if time, 4 = last)
    python scripts/09_gpu_batch.py run --all --commit   # train everything, commit and push the small result files
    python scripts/09_gpu_batch.py status               # validation results of the finished runs
    python scripts/09_gpu_batch.py upload v2_kids_plus_all   # publish chosen models on the Hugging Face Hub

The list of runs, the rule for choosing the model that is shown in the demo, and the thresholds are fixed
in docs/V2_PROTOCOL.md, which was committed before any of these models was trained. This script only
carries that list out. It never changes or uploads the two published first models.

What a run does:
  1. scripts/02_train.py with --no-test: the model is trained and its VALIDATION results are stored.
     No test result is computed or shown on the training machine.
  2. The small files that the analysis needs are copied to results/v2/<name>/ (git): the model's settings,
     its score for every X-ray (130 KB) and the list of adult X-rays it was trained on.
     Weights (45 MB) stay in artifacts/<name>/ and go to the Hub with "upload".
A finished run is skipped, so the script can be stopped and started again at any time.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pneumoshift import config as C  # noqa: E402

V2 = C.RESULTS_DIR / "v2"
LOGS = ROOT / "logs" / "gpu"
PROTECTED = ("kids_only", "kids_plus_adults")
KEEP_PREDICTIONS = ("v2_kids_plus_all",)        # the headline model of docs/V2_PROTOCOL.md
ALL_ADULTS = "--rsna-n 16000"            # more than the 15,142 adult training X-rays, so: all of them
COMMON = "--select last --no-test --require-gpu --workers 0"


def plan() -> list[dict]:
    """The fixed list of runs (docs/V2_PROTOCOL.md). Order matters: the most important ones come first.

    One recipe for every run: ResNet-18, 224 px, 6 epochs, the last epoch is kept. Only the training
    X-rays and the random seed differ, so differences between runs can be put down to those.
    """
    runs = []

    def add(stage, name, flags, why):
        runs.append(dict(stage=stage, name=name, flags=flags, why=why))

    def kids_plus(n, seed):
        tail = "" if seed == C.SEED else f"_s{seed}"
        if n == 0:
            return f"v2_kids_plus_0{tail}", f"--seed {seed}", "children only (the tutorial recipe)"
        if n == "all":
            return f"v2_kids_plus_all{tail}", f"{ALL_ADULTS} --val adults --seed {seed}", "REPAIRED MODEL: children + all 15,142 adult training X-rays"
        return f"v2_kids_plus_{n}{tail}", f"--rsna-n {n} --val adults --seed {seed}", f"children + {n:,} adult X-rays"

    def adults_only(seed):
        tail = "" if seed == C.SEED else f"_s{seed}"
        return f"v2_adults_all{tail}", f"--no-kids {ALL_ADULTS} --seed {seed}", "adults only, all 15,142 (do the children's X-rays help or hurt?)"

    # Stage 1 (must): the repaired model, its same-recipe comparison points, and the repair curve, seed 2026
    for n in ("all", 6000, 0, 300, 1000, 3000):
        add(1, *kids_plus(n, C.SEED))
    add(1, *adults_only(C.SEED))
    # Stage 2 (should): is it luck? Two more seeds for the three points the main claim rests on
    for seed in (2027, 2028):
        for n in ("all", 6000, 0):
            add(2, *kids_plus(n, seed))
    # Stage 3 (if time): two more seeds for the rest of the curve and the adults-only model
    for seed in (2027, 2028):
        for n in (300, 1000, 3000):
            add(3, *kids_plus(n, seed))
        add(3, *adults_only(seed))
    # Stage 4 (last): would a bigger network help? Reported as a side note, never the model in the demo
    add(4, "v2_kids_plus_all_r50", f"{ALL_ADULTS} --val adults --seed {C.SEED} --arch resnet50.tv_in1k", "side note: ResNet-50 instead of ResNet-18")
    return runs


def is_done(name: str) -> bool:
    return (V2 / name / "config.json").exists() and (V2 / name / "logits.npz").exists()


def pack(name: str) -> None:
    """artifacts/<name>/ -> results/v2/<name>/: settings, scores for every X-ray, training indices. No weights."""
    src, dst = C.ARTIFACTS_DIR / name, V2 / name
    dst.mkdir(parents=True, exist_ok=True)
    kermany = np.load(src / "preds_kermany.npz")["logits"].astype(np.float32)
    rsna = np.load(src / "preds_rsna.npz")["logits"].astype(np.float32)
    np.savez_compressed(dst / "logits.npz", kermany=kermany, rsna=rsna)
    if (src / "rsna_train_idx.npy").exists():
        np.save(dst / "rsna_train_idx.npy", np.load(src / "rsna_train_idx.npy").astype(np.int32))
    shutil.copy2(src / "config.json", dst / "config.json")          # written last: its presence means "done"
    # The per-image files in artifacts/ also hold 512 numbers per X-ray (55 MB per model). Only the headline
    # model needs them later (for its "unlike my training data" check), so the others give the disk space back.
    if name not in KEEP_PREDICTIONS:
        for f in ("preds_kermany.npz", "preds_rsna.npz"):
            (src / f).unlink(missing_ok=True)


def train(run: dict, extra: str = "") -> bool:
    name = run["name"]
    LOGS.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", str(ROOT / "scripts" / "02_train.py"), "--name", name] + (run["flags"] + " " + COMMON + " " + extra).split()
    print(f"\n=== {name}: {run['why']}\n    {' '.join(cmd[2:])}", flush=True)
    t = time.time()
    with open(LOGS / f"{name}.log", "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace",
                                env=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1"))
        for line in proc.stdout:
            log.write(line); log.flush()
            print("    " + line.rstrip(), flush=True)
        code = proc.wait()
    print(f"=== {name}: {'finished' if code == 0 else f'FAILED with exit code {code}'} after {(time.time() - t) / 60:.1f} min", flush=True)
    return code == 0


def git(*args: str) -> bool:
    ident = ["-c", "user.name=PneumoShift GPU runs", "-c", "user.email=pneumoshift@users.noreply.github.com"]
    r = subprocess.run(["git", *ident, *args], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    git {' '.join(args[:2])} failed: {(r.stderr or r.stdout).strip().splitlines()[-1:]}", flush=True)
    return r.returncode == 0


def commit_and_push(message: str) -> None:
    """Only the small result files of v2 runs are added. Nothing else in the working folder is touched."""
    paths = ["results/v2"] + [str(p.relative_to(ROOT)).replace("\\", "/") for p in sorted(C.RESULTS_DIR.glob("val_v2_*.json"))]
    if not git("add", "--", *paths):
        return
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0:
        return                                                  # nothing new
    if git("commit", "-m", message) and not git("push", "origin", "HEAD"):
        print("    the results are committed locally but NOT pushed. Push them by hand: git push origin HEAD", flush=True)


def write_status(rows: list[dict]) -> None:
    V2.mkdir(parents=True, exist_ok=True)
    (V2 / "STATUS.json").write_text(json.dumps(dict(
        note="Written by scripts/09_gpu_batch.py. Validation results only, no test image is involved.", runs=rows), indent=1))


def status_rows() -> list[dict]:
    rows = []
    for run in plan():
        row = dict(name=run["name"], stage=run["stage"], status="done" if is_done(run["name"]) else "open")
        if row["status"] == "done":
            cfg = json.loads((V2 / run["name"] / "config.json").read_text())
            val = cfg.get("validation", {})
            row.update(minutes=round(cfg.get("train_minutes", 0), 1), best_epoch=cfg.get("best_epoch"), device=cfg.get("device"),
                       **{f"val_auroc_{k}": round(v["auroc"], 4) for k, v in val.items()},
                       val_adult_specificity_at_threshold=round(val.get("adults", {}).get("specificity", float("nan")), 4),
                       val_adult_sensitivity_at_threshold=round(val.get("adults", {}).get("sensitivity", float("nan")), 4))
        elif (LOGS / f"{run['name']}.log").exists():
            row["status"] = "failed or interrupted (see logs/gpu/)"
        rows.append(row)
    return rows


def show(rows: list[dict]) -> None:
    print(f"\n{'run':28s} {'stage':5s} {'status':10s} {'min':>5s}  validation AUROC: adults  standing(PA)  bedside(AP)  children")
    for r in rows:
        if r["status"] == "done":
            print(f"{r['name']:28s} {r['stage']:<5d} {'done':10s} {r['minutes']:5.1f}  {'':18s}{r.get('val_auroc_adults', float('nan')):.4f}      "
                  f"{r.get('val_auroc_adults_PA', float('nan')):.4f}       {r.get('val_auroc_adults_AP', float('nan')):.4f}    {r.get('val_auroc_children', float('nan')):.4f}")
        else:
            print(f"{r['name']:28s} {r['stage']:<5d} {r['status']}")
    print("\nThese are VALIDATION results. Test results are computed once, at the end, by the analysis (not on this machine).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["list", "smoke", "run", "status", "upload"])
    ap.add_argument("names", nargs="*", help="model names (for upload, or to run only these)")
    ap.add_argument("--stage", type=int, action="append", help="run only this stage (may be given several times)")
    ap.add_argument("--all", action="store_true", help="run every stage")
    ap.add_argument("--commit", action="store_true", help="after each run, commit and push the small result files")
    ap.add_argument("--extra", default="", help="extra flags for 02_train.py, for example \"--amp\" or \"--batch-size 16\"")
    ap.add_argument("--cpu", action="store_true", help="allow training without a GPU (slow: 20 to 60 minutes per run)")
    args = ap.parse_args()
    try:        # a Windows console cannot print every character a download progress bar may contain
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    if args.cpu:
        COMMON = COMMON.replace("--require-gpu", "").replace("--workers 0", "--workers 2")
    runs = plan()
    assert not any(r["name"] in PROTECTED for r in runs) and len({r["name"] for r in runs}) == len(runs)

    if args.what == "list":
        for r in runs:
            print(f"stage {r['stage']}  {'done' if is_done(r['name']) else 'open':5s} {r['name']:28s} {r['why']}")

    elif args.what == "smoke":
        name = "v2_smoke"
        ok = train(dict(name=name, flags="--rsna-n 300 --val adults --epochs 1 --overwrite", why="test of the whole chain (not a result)"), args.extra)
        if ok:
            try:
                pack(name)
                z = np.load(V2 / name / "logits.npz")
                ok = len(z["kermany"]) == 5856 and len(z["rsna"]) == 26684 and bool(np.isfinite(z["rsna"]).all())
            except Exception as e:
                print(f"    packing failed: {type(e).__name__}: {e}")
                ok = False
        for p in (C.ARTIFACTS_DIR / name, V2 / name):
            shutil.rmtree(p, ignore_errors=True)
        (C.RESULTS_DIR / f"val_{name}.json").unlink(missing_ok=True)
        print("\nSMOKE TEST OK: training, validation, scores for all X-rays and packing work on this machine."
              if ok else "\nSMOKE TEST FAILED: see the lines above and logs/gpu/v2_smoke.log")
        sys.exit(0 if ok else 1)

    elif args.what == "run":
        if args.names:
            todo = [r for r in runs if r["name"] in args.names]
            unknown = set(args.names) - {r["name"] for r in runs}
            if unknown:
                sys.exit(f"not in the plan: {sorted(unknown)} (see: python scripts/09_gpu_batch.py list)")
        elif args.all:
            todo = runs
        elif args.stage:
            todo = [r for r in runs if r["stage"] in args.stage]
        else:
            sys.exit("say what to run: --stage N, --all, or model names")
        failed = []
        for run in todo:
            name = run["name"]
            if is_done(name):
                print(f"=== {name}: already done, skipped", flush=True)
                continue
            trained = (C.ARTIFACTS_DIR / name / "model.safetensors").exists() and (C.ARTIFACTS_DIR / name / "preds_rsna.npz").exists()
            if not trained:
                shutil.rmtree(C.ARTIFACTS_DIR / name, ignore_errors=True)       # leftovers of an interrupted run
                if not train(run, args.extra):
                    failed.append(name)
                    write_status(status_rows())
                    continue
            pack(name)
            write_status(status_rows())
            if args.commit:
                commit_and_push(f"v2 run {name}: settings, validation results and scores (no test results)")
        show(status_rows())
        if failed:
            sys.exit(f"\nFAILED runs: {failed}. Their logs are in logs/gpu/. The other runs are complete.")
        print("\nALL REQUESTED RUNS ARE DONE.")

    elif args.what == "status":
        show(status_rows())

    elif args.what == "upload":
        if not args.names:
            sys.exit("name the models to upload, for example: python scripts/09_gpu_batch.py upload v2_kids_plus_all")
        for name in args.names:
            if name in PROTECTED or not name.startswith("v2_"):
                sys.exit(f"'{name}' is not a v2 model. This command uploads v2 models only.")
            if not (C.ARTIFACTS_DIR / name / "model.safetensors").exists():
                sys.exit(f"artifacts/{name}/ has no trained model")
        for name in args.names:
            r = subprocess.run([sys.executable, str(ROOT / "scripts" / "hub_models.py"), "push", name], cwd=ROOT)
            print(f"=== {name}: {'uploaded' if r.returncode == 0 else 'UPLOAD FAILED (logged in? hf auth login)'}", flush=True)
