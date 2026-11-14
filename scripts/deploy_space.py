"""Publish the demo and the models on the Hugging Face Hub (needs `huggingface-cli login` or HF_TOKEN).

    python scripts/deploy_space.py space RL8596/pneumoshift                       # app/ + src/pneumoshift/ -> a Space
    python scripts/deploy_space.py model artifacts/kids_only RL8596/pneumoshift-kids-only
    python scripts/deploy_space.py model artifacts/kids_plus_adults RL8596/pneumoshift-kids-plus-adults

Add --dry-run to see which files would be uploaded without contacting the Hub.
Upload the models first: the Space downloads them when it starts.
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_FILES = ("model.safetensors", "config.json", "ood.npz")          # nothing else leaves the model folder
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".gradio", "flagged", ".DS_Store", "*.egg-info")


def stage_space(dest: Path) -> Path:
    """Build the Space layout in dest: app.py, requirements.txt, README.md, examples/ and pneumoshift/."""
    shutil.copytree(REPO_ROOT / "app", dest, ignore=IGNORE, dirs_exist_ok=True)
    shutil.copytree(REPO_ROOT / "src" / "pneumoshift", dest / "pneumoshift", ignore=IGNORE, dirs_exist_ok=True)
    return dest


def model_card(cfg: dict, repo_id: str) -> str:
    """README.md of a model repo, written from the model's own config. No number is added by hand."""
    name = cfg.get("display_name") or cfg.get("name") or repo_id.split("/")[-1]
    data = [str(s) for s in (cfg.get("train_data") or [])]
    lines = [
        "---", "license: mit", "library_name: timm", "pipeline_tag: image-classification",
        "tags:", "- medical-imaging", "- chest-x-ray", "- pneumonia", "---", "",
        f"# PneumoShift: {name}", "",
        "**Research model. Not a medical device. Not for diagnosis.**", "",
        f"A `{cfg.get('arch', 'unknown')}` image classifier with one output: the chance that a frontal chest X-ray "
        "shows pneumonia. Part of a student project that asks whether a model trained on children still works on adults.", "",
        "## Training data", "",
        *([f"- {d}" for d in data] or ["- not recorded"]), "",
        "Children: Kermany et al. 2018 (ages 1 to 5, one hospital in Guangzhou, China, CC BY 4.0). "
        "Adults, if listed above: RSNA Pneumonia Detection Challenge 2018 (USA).", "",
        "## Intended use", "",
        "Flag chest X-rays for priority review by a clinician, in a research or teaching setting. "
        "The model must not be used to diagnose or treat anyone.", "",
        "## How to use", "",
        "```python", "from pneumoshift.inference import Predictor",
        f'r = Predictor("{repo_id}").predict(pil_image)   # probability, flagged, heatmap, ood_level, ...', "```", "",
        f"- Probability = sigmoid(logit / {cfg.get('temperature', 1.0):.4g}) (temperature scaling).",
        f"- An X-ray is flagged when the probability is at least {cfg.get('threshold', 0.5):.4g}"
        + (f" ({cfg['threshold_rule']})." if cfg.get("threshold_rule") else "."),
        "- `ood.npz`, if present, holds the statistics for the \"unlike my training data\" warning.", "",
        "## Test results", "",
    ]
    metrics = cfg.get("metrics") if isinstance(cfg.get("metrics"), dict) else {}
    if metrics:
        lines += ["| Test X-rays (never used for training) | Number | Sensitivity | Specificity | AUROC |", "|---|---|---|---|---|"]
        for group, m in metrics.items():
            f = lambda k, spec: format(m[k], spec) if isinstance(m.get(k), (int, float)) else "n/a"  # noqa: E731
            lines.append(f"| {group} | {f('n', 'd')} | {f('sensitivity', '.3f')} | {f('specificity', '.3f')} | {f('auroc', '.3f')} |")
    else:
        lines.append("Not measured yet.")
    lines += [
        "", "## Limitations", "",
        "- It only learned from the patient groups and hospitals listed under training data. For other age groups, "
        "hospitals or X-ray machines it can be confidently wrong.",
        "- It knows two answers, pneumonia or not, and does not recognise other findings.",
        "- The out-of-distribution warning catches many unusual X-rays, not all of them.",
        "- Not checked by any authority.", ""]
    return "\n".join(lines)


def stage_model(folder: Path, repo_id: str, dest: Path) -> Path:
    folder = Path(folder)
    for name in ("model.safetensors", "config.json"):
        if not (folder / name).exists():
            sys.exit(f"{folder / name} is missing")
    for name in MODEL_FILES:
        if (folder / name).exists():
            shutil.copy2(folder / name, dest / name)
    cfg = json.loads((folder / "config.json").read_text())
    (dest / "README.md").write_text(model_card(cfg, repo_id))
    return dest


def listing(root: Path) -> str:
    return "\n".join(f"  {p.relative_to(root)}  ({p.stat().st_size:,} bytes)" for p in sorted(root.rglob("*")) if p.is_file())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    sp = sub.add_parser("space", help="upload app/ and src/pneumoshift/ to a Space")
    sp.add_argument("repo_id", help="for example RL8596/pneumoshift")
    mp = sub.add_parser("model", help="upload one model folder to a model repo")
    mp.add_argument("folder", type=Path, help="for example artifacts/kids_only")
    mp.add_argument("repo_id", help="for example RL8596/pneumoshift-kids-only")
    for p in (sp, mp):
        p.add_argument("--private", action="store_true", help="create the repo as private")
        p.add_argument("--dry-run", action="store_true", help="only list the files, upload nothing")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        staged = stage_space(Path(tmp)) if args.what == "space" else stage_model(args.folder, args.repo_id, Path(tmp))
        print(f"Files for the {args.what} repo {args.repo_id}:\n{listing(staged)}")
        if args.dry_run:
            sys.exit("Dry run: nothing was uploaded.")

        from huggingface_hub import HfApi

        api = HfApi()
        if args.what == "space":
            api.create_repo(args.repo_id, repo_type="space", space_sdk="gradio", exist_ok=True, private=args.private)
        else:
            api.create_repo(args.repo_id, repo_type="model", exist_ok=True, private=args.private)
        api.upload_folder(folder_path=str(staged), repo_id=args.repo_id, repo_type=args.what,
                          commit_message=f"Upload {args.what} from scripts/deploy_space.py")
        kind = "spaces/" if args.what == "space" else ""
        print(f"Done: https://huggingface.co/{kind}{args.repo_id}")
