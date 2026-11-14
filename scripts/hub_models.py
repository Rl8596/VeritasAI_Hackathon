"""Move trained models between artifacts/ and the Hugging Face Hub.

    python scripts/hub_models.py push v2_kids_plus_all   # weights, config, model card, analysis files (names are required;
                                                         # the two published first models are refused)
    python scripts/hub_models.py pull               # every published model -> artifacts/<name>/

`pull` is what a fresh clone needs: artifacts/ is not in git. The analysis files
(per-image predictions and embeddings) let steps 05 to 07 run without the network.
Pushing needs a write token (HF_TOKEN or `huggingface-cli login`); pulling needs none.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pneumoshift import config as C  # noqa: E402

USER = os.environ.get("PNEUMOSHIFT_HF_USER", "RL8596")
NAMES = ["kids_only", "kids_plus_adults", "kids_plus_300", "kids_plus_1000", "adults_only"]
ANALYSIS = ["preds_kermany.npz", "preds_rsna.npz", "rsna_train_idx.npy"]


def repo(name: str) -> str:
    return f"{USER}/pneumoshift-{name.replace('_', '-')}"


PROTECTED = ("kids_only", "kids_plus_adults")     # the published first models; the live demo loads them from the Hub

if __name__ == "__main__":
    what, names = sys.argv[1], sys.argv[2:] or NAMES
    if what == "push":
        if not sys.argv[2:]:
            sys.exit("push needs model names, for example: python scripts/hub_models.py push v2_kids_plus_all")
        blocked = [n for n in names if n in PROTECTED]
        if blocked and os.environ.get("PNEUMOSHIFT_ALLOW_PROTECTED") != "1":
            sys.exit(f"{blocked} are the published first models. Pushing would change what the live demo loads. "
                     "Train new models under new names instead.")
    from huggingface_hub import HfApi, snapshot_download

    for name in names:
        folder = C.ARTIFACTS_DIR / name
        if what == "push":
            subprocess.run([sys.executable, str(ROOT / "scripts" / "deploy_space.py"), "model", str(folder), repo(name)], check=True)
            for f in ANALYSIS:
                if (folder / f).exists():
                    HfApi().upload_file(path_or_fileobj=str(folder / f), path_in_repo=f"analysis/{f}", repo_id=repo(name))
            print("pushed", repo(name))
        else:
            try:
                snapshot_download(repo(name), local_dir=str(folder))
            except Exception as e:
                print(f"skipped {repo(name)}: {type(e).__name__}"); continue
            for f in (folder / "analysis").glob("*"):
                shutil.move(str(f), str(folder / f.name))
            print("pulled", repo(name), "->", folder)
