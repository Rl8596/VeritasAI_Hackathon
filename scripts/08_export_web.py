"""Export the trained models for the browser demo in web/ (no server, the model runs on the visitor's device).

    python scripts/08_export_web.py                    # export both models, check them, write the reference file
    python scripts/08_export_web.py push               # upload the exported files to the model repos (write token)
    python scripts/08_export_web.py space RL8596/pneumoshift   # upload web/ as a static Space (write token)

For every model this writes three files to artifacts/<name>/onnx/:

    model.onnx   the network. One run returns the pneumonia logit, the 512 embedding numbers and the heatmap.
    ood.bin      the numbers of the "unlike my training data" check (float32: class means, precision matrix,
                 sorted validation scores), read by web/app.js
    meta.json    threshold, temperature, test results and the layout of ood.bin

The heatmap is computed inside the exported network. For a ResNet (average pooling, then one linear layer)
the gradient of the logit with respect to a feature-map channel is the same at every position: the weight of
that channel divided by the number of positions. So Grad-CAM needs no backward pass here, and the exported
heatmap is the same map that src/pneumoshift/cam.py computes with autograd. This script checks that claim:
it runs every example picture through the PyTorch code path of the Gradio app and through the exported file
and stops if they disagree.

It also writes tests/web/reference.json (what the Python code path returns for each test picture). The browser
test tests/web/parity.mjs compares the page against that file.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pneumoshift import config as C  # noqa: E402
from pneumoshift.inference import Predictor  # noqa: E402

USER = "RL8596"
NAMES = ["kids_only", "kids_plus_adults"]
WEB = ROOT / "web"
TESTS = ROOT / "tests" / "web"
OPSET = 17
MAX_DIFF = dict(logit=1e-3, embedding=1e-3, cam=1e-3)      # largest allowed difference, PyTorch vs exported file


def repo(name: str) -> str:
    return f"{USER}/pneumoshift-{name.replace('_', '-')}"


def out_dir(name: str) -> Path:
    return C.ARTIFACTS_DIR / name / "onnx"


class WebModel(nn.Module):
    """model input [B, 3, H, W] -> (logit [B], embedding [B, D], heatmap [B, H, W] in 0..1)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        m = self.model
        feats = m.forward_features(x)                               # [B, C, h, w]
        emb = m.forward_head(feats, pre_logits=True)                # [B, C], average over the h*w positions
        fc = m.get_classifier()
        logit = fc(emb).squeeze(1)
        weights = fc.weight.view(1, -1, 1, 1) / (feats.shape[2] * feats.shape[3])   # = Grad-CAM weights, see docstring
        cam = F.relu((weights * feats).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
        return logit, emb, cam


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pictures(tmp: Path) -> list[dict]:
    """The example pictures, plus three made from them that exercise the resizing code:
    a large non-square picture, a small one, and a colour picture."""
    examples = json.loads((ROOT / "app" / "examples" / "examples.json").read_text())["examples"]
    pictures = [dict(file=f"examples/{e['file']}", path=ROOT / "app" / "examples" / e["file"]) for e in examples]
    tmp.mkdir(parents=True, exist_ok=True)
    base = Image.open(ROOT / "app" / "examples" / "adult_lung_opacity_1.png").convert("L")
    made = {
        "resize_large_1500x1100.png": base.resize((1500, 1100), Image.Resampling.BICUBIC),
        "resize_small_180x140.png": base.resize((180, 140), Image.Resampling.BICUBIC),
        "resize_colour_701x903.png": Image.merge("RGB", (
            base.resize((701, 903), Image.Resampling.BICUBIC),
            base.resize((701, 903), Image.Resampling.BICUBIC).point(lambda v: v * 3 // 4),
            base.resize((701, 903), Image.Resampling.BICUBIC).point(lambda v: 255 - v))),
    }
    for name, img in made.items():
        img.save(tmp / name)
        pictures.append(dict(file=f"pictures/{name}", path=tmp / name))
    return pictures


def export(name: str, pictures: list[dict]) -> dict:
    import onnxruntime as ort

    local = C.ARTIFACTS_DIR / name
    predictor = Predictor(local if (local / "model.safetensors").exists() else repo(name))
    cfg, dest = predictor.config, out_dir(name)
    dest.mkdir(parents=True, exist_ok=True)

    # 1. the network
    wrapped = WebModel(predictor.model).eval()
    size = int(cfg.get("img_size", C.IMG_SIZE))
    torch.onnx.export(wrapped, (torch.zeros(1, 3, size, size),), str(dest / "model.onnx"), dynamo=False,
                      input_names=["input"], output_names=["logit", "embedding", "cam"], opset_version=OPSET,
                      do_constant_folding=True)

    # 2. the numbers of the familiarity check
    ood, layout = predictor.ood, None
    if ood is not None:
        parts = [ood.means.astype("<f4"), ood.precision.astype("<f4"), np.asarray(ood.ref_scores, dtype="<f4")]
        (dest / "ood.bin").write_bytes(b"".join(p.tobytes() for p in parts))
        layout = dict(n_classes=int(ood.means.shape[0]), dim=int(ood.means.shape[1]), n_ref=int(len(ood.ref_scores)),
                      p95=float(ood.thresholds["p95"]), p99=float(ood.thresholds["p99"]), bytes=(dest / "ood.bin").stat().st_size,
                      order="float32 little-endian: means [n_classes, dim], precision [dim, dim], ref_scores [n_ref] (sorted)")

    # 3. check the exported file against the PyTorch code path of the Gradio app
    session = ort.InferenceSession(str(dest / "model.onnx"), providers=["CPUExecutionProvider"])
    reference, worst = {}, dict(logit=0.0, embedding=0.0, cam=0.0)
    for pic in pictures:
        r = predictor.predict(Image.open(pic["path"]))
        from pneumoshift.inference import to_pil
        from pneumoshift.preprocess import pil_to_model_input

        x, gray = pil_to_model_input(to_pil(Image.open(pic["path"])))
        logit, emb, cam = session.run(None, {"input": x.numpy()})
        with torch.no_grad():
            emb_torch = predictor.model.forward_head(predictor.model.forward_features(x), pre_logits=True).numpy()
        worst["logit"] = max(worst["logit"], abs(float(logit[0]) - r["logit"]))
        worst["embedding"] = max(worst["embedding"], float(np.abs(emb - emb_torch).max()))
        worst["cam"] = max(worst["cam"], float(np.abs(cam[0] - r["cam"]).max()))
        hottest = np.unravel_index(int(np.argmax(r["cam"])), r["cam"].shape)
        reference[pic["file"]] = dict(
            logit=r["logit"], probability=r["probability"], flagged=r["flagged"],
            ood_score=r["ood_score"], ood_percentile=r["ood_percentile"], ood_level=r["ood_level"],
            input_sha256=hashlib.sha256(np.ascontiguousarray(gray).tobytes()).hexdigest(),
            cam_hottest_yx=[int(hottest[0]), int(hottest[1])],
            cam_mean=float(r["cam"].mean()),
            heatmap_mean_rgb=[float(v) for v in r["heatmap"].reshape(-1, 3).mean(0)],
        )
    for key, limit in MAX_DIFF.items():
        if worst[key] > limit:
            sys.exit(f"{name}: the exported model disagrees with PyTorch ({key}: {worst[key]:.2e} > {limit:.0e})")

    # 4. what the page needs to know about the model
    keep = ("arch", "img_size", "store_size", "class_names", "temperature", "threshold", "threshold_rule", "n_train",
            "name", "display_name", "train_data", "train_groups", "val_data", "metrics", "ood_summary")
    meta = dict(
        format="pneumoshift-web-1",
        config={k: cfg[k] for k in keep if k in cfg},
        input=dict(name="input", shape=[1, 3, size, size], mean=list(C.IMAGENET_MEAN), std=list(C.IMAGENET_STD)),
        outputs=["logit", "embedding", "cam"],
        files={"model.onnx": dict(bytes=(dest / "model.onnx").stat().st_size, sha256=sha256(dest / "model.onnx"))},
        ood=layout,
        check=dict(pictures=len(pictures), max_abs_difference_to_pytorch=worst,
                   note="logit, embedding and heatmap of the exported file compared with the PyTorch model"),
    )
    if layout:
        meta["files"]["ood.bin"] = dict(bytes=layout["bytes"], sha256=sha256(dest / "ood.bin"))
    (dest / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{name}: {dest}  model.onnx {meta['files']['model.onnx']['bytes'] / 1e6:.1f} MB, "
          f"largest difference to PyTorch: " + ", ".join(f"{k} {v:.1e}" for k, v in worst.items()))
    return reference


def inferno_table() -> list[list[float]]:
    """The 256 colours of matplotlib's 'inferno', as used by src/pneumoshift/cam.py."""
    from matplotlib import colormaps

    return [[round(float(c), 5) for c in rgb[:3]] for rgb in colormaps["inferno"](np.arange(256))]


def stage_space(dest: Path) -> Path:
    shutil.copytree(WEB, dest, dirs_exist_ok=True, ignore=shutil.ignore_patterns("models", ".DS_Store"))
    (dest / "README.md").write_text((WEB / "SPACE_README.md").read_text())
    (dest / "SPACE_README.md").unlink()
    return dest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", nargs="?", default="export", choices=["export", "push", "space"])
    ap.add_argument("repo_id", nargs="?", default=f"{USER}/pneumoshift", help="Space repo id (only for 'space')")
    ap.add_argument("--dry-run", action="store_true", help="list what would be uploaded, upload nothing")
    args = ap.parse_args()

    if args.what == "export":
        pictures = test_pictures(TESTS / "pictures")
        reference = {name: export(name, pictures) for name in NAMES}
        (TESTS / "reference.json").write_text(json.dumps(dict(
            note="What the Python code path (src/pneumoshift/inference.py) returns for each test picture. "
                 "Written by scripts/08_export_web.py, compared with the browser by tests/web/parity.mjs.",
            models=reference), indent=1))
        shutil.copytree(ROOT / "app" / "examples", WEB / "examples", dirs_exist_ok=True)
        (WEB / "inferno.json").write_text(json.dumps(inferno_table(), separators=(",", ":")))
        # a local copy of the model files, so the page can be tested without the Hub (web/models/ is not in git)
        for name in NAMES:
            shutil.copytree(out_dir(name), WEB / "models" / name, dirs_exist_ok=True)
        print(f"wrote {TESTS / 'reference.json'}, web/examples/, web/inferno.json, web/models/")

    elif args.what == "push":
        from huggingface_hub import HfApi

        for name in NAMES:
            files = sorted(out_dir(name).glob("*"))
            if not (out_dir(name) / "model.onnx").exists():
                sys.exit(f"{out_dir(name)} is missing: run this script without arguments first")
            print(repo(name), [f"onnx/{f.name} ({f.stat().st_size:,} bytes)" for f in files])
            if not args.dry_run:
                HfApi().upload_folder(folder_path=str(out_dir(name)), path_in_repo="onnx", repo_id=repo(name),
                                      commit_message="Browser version of the model (scripts/08_export_web.py)")
                print(f"  -> https://huggingface.co/{repo(name)}/tree/main/onnx")

    else:
        import tempfile

        from huggingface_hub import HfApi

        with tempfile.TemporaryDirectory() as tmp:
            staged = stage_space(Path(tmp))
            for p in sorted(staged.rglob("*")):
                if p.is_file():
                    print(f"  {p.relative_to(staged)}  ({p.stat().st_size:,} bytes)")
            if args.dry_run:
                sys.exit("Dry run: nothing was uploaded.")
            api = HfApi()
            api.create_repo(args.repo_id, repo_type="space", space_sdk="static", exist_ok=True)
            api.upload_folder(folder_path=str(staged), repo_id=args.repo_id, repo_type="space",
                              commit_message="Browser demo from scripts/08_export_web.py")
            print(f"Done: https://huggingface.co/spaces/{args.repo_id}")
