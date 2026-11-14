"""PneumoShift demo: a second look at a chest X-ray, with warnings when the model is out of its depth.

Run locally from the repo:    python app/app.py
    PNEUMOSHIFT_DEV=1         also load the untrained placeholder model artifacts/_smoke (development only)
    PNEUMOSHIFT_THREADS=1     limit PyTorch to this many CPU threads (default: PyTorch decides)

On Hugging Face Spaces this file sits at the repo root, next to requirements.txt,
README.md, examples/ and a copy of the pneumoshift/ package (see scripts/deploy_space.py).
"""
import hashlib
import html
import json
import os
import sys
import threading
import time
import traceback
from collections import OrderedDict
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
for _candidate in (APP_DIR, APP_DIR.parent / "src"):          # Spaces layout, then repo layout
    if (_candidate / "pneumoshift" / "__init__.py").exists():
        sys.path.insert(0, str(_candidate))
        break

import gradio as gr  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from pneumoshift import config as C  # noqa: E402
from pneumoshift.inference import Predictor, to_pil  # noqa: E402

# ----------------------------------------------------------------------------
# Models. Each one is loaded from the local folder if it exists, otherwise from
# the Hugging Face Hub. Models that cannot be loaded are skipped; the app works
# with whichever are available.
# ----------------------------------------------------------------------------
MODEL_SOURCES = [
    dict(
        display_name="Children-only model",
        repo=os.environ.get("PNEUMOSHIFT_KIDS_ONLY_REPO", "RL8596/pneumoshift-kids-only"),
        local=C.ARTIFACTS_DIR / "kids_only",                  # artifacts/kids_only
    ),
    dict(
        display_name="Children + adults model",
        repo=os.environ.get("PNEUMOSHIFT_KIDS_PLUS_ADULTS_REPO", "RL8596/pneumoshift-kids-plus-adults"),
        local=C.ARTIFACTS_DIR / "kids_plus_adults",           # artifacts/kids_plus_adults
    ),
]
if os.environ.get("PNEUMOSHIFT_DEV") == "1":
    MODEL_SOURCES.append(dict(
        display_name="Placeholder model (development only, untrained: its results mean nothing)",
        repo=None,
        local=C.ARTIFACTS_DIR / "_smoke",                     # artifacts/_smoke
    ))

EXAMPLES_DIR = APP_DIR / "examples"
MAX_UPLOAD = "30mb"

AGE_CHOICES = [                       # (what the user sees, internal value)
    ("Not sure", "unknown"),
    ("Child, 1 to 5 years", "child_1_5"),
    ("Child or teenager, 6 to 17 years", "child_6_17"),
    ("Adult, 18 years or older", "adult"),
]
# which entry of a model's `train_groups` covers which choice. Nobody trained on ages 6 to 17.
AGE_TO_GROUP = {"child_1_5": "children", "child_6_17": None, "adult": "adults"}
AGE_WORDS = {"child_1_5": "children aged 1 to 5", "child_6_17": "children or teenagers aged 6 to 17", "adult": "adults"}
GROUP_WORDS = {"children": "children aged 1 to 5", "adults": "adults"}

if os.environ.get("PNEUMOSHIFT_THREADS"):
    torch.set_num_threads(int(os.environ["PNEUMOSHIFT_THREADS"]))


def log(message: str) -> None:
    print(f"[pneumoshift] {message}", flush=True)


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_models() -> list[dict]:
    models = []
    for src in MODEL_SOURCES:
        local = Path(src["local"]) if src.get("local") else None
        if local and (local / "config.json").exists() and (local / "model.safetensors").exists():
            where = str(local)
        elif src.get("repo"):
            where = src["repo"]
        else:
            continue
        try:
            t = time.time()
            predictor = Predictor(where)
            predictor.predict(Image.new("L", (C.STORE_SIZE, C.STORE_SIZE), 0))      # warm-up, and an early test
            cfg = predictor.config
            models.append(dict(predictor=predictor, config=cfg, source=where,
                               name=str(cfg.get("display_name") or src["display_name"])))
            log(f"loaded '{models[-1]['name']}' from {where} in {time.time() - t:.1f}s "
                f"(familiarity check: {'yes' if predictor.ood is not None else 'no ood.npz'}, "
                f"test results: {'yes' if cfg.get('metrics') else 'none in config'})")
        except Exception as e:                # a missing or broken model must not stop the demo
            log(f"skipped '{src['display_name']}' ({where}): {type(e).__name__}: {str(e).splitlines()[0][:200]}")
    return models


def load_examples() -> list[dict]:
    """Entries of examples/examples.json whose picture exists, each with its path and a content hash."""
    try:
        manifest = json.loads((EXAMPLES_DIR / "examples.json").read_text())
        examples = []
        for e in manifest.get("examples", []):
            path = EXAMPLES_DIR / str(e.get("file", ""))
            if path.is_file():
                examples.append(dict(e, path=str(path), sha=_sha(path)))
        return examples
    except Exception as e:
        log(f"no examples ({type(e).__name__}: {e})")
        return []


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


MODELS = load_models()
EXAMPLES = load_examples()
EXAMPLE_BY_SHA = {e["sha"]: e for e in EXAMPLES}
log(f"{len(MODELS)} model(s), {len(EXAMPLES)} example(s), {torch.get_num_threads()} CPU thread(s)")


# ----------------------------------------------------------------------------
# Predictions (cached per uploaded file, so changing the age group is instant)
# ----------------------------------------------------------------------------
_cache: "OrderedDict[str, dict]" = OrderedDict()
_cache_lock = threading.Lock()


def run_models(path: str) -> dict:
    """-> {"example": the examples.json entry if this picture is one of our examples, else None,
           "results": one Predictor.predict dict (or None if that model failed) per model}.
    Raises ValueError with a readable message if the file is not a usable picture."""
    with _cache_lock:
        if path in _cache:
            _cache.move_to_end(path)
            return _cache[path]
    image = to_pil(path)
    results = []
    for m in MODELS:
        try:
            results.append(m["predictor"].predict(image))
        except Exception:
            log(f"prediction failed for '{m['name']}':\n{traceback.format_exc()}")
            results.append(None)
    try:
        example = EXAMPLE_BY_SHA.get(_sha(path))
    except OSError:
        example = None
    entry = dict(example=example, results=results)
    with _cache_lock:
        _cache[path] = entry
        while len(_cache) > 32:
            _cache.popitem(last=False)
    return entry


# ----------------------------------------------------------------------------
# Words and HTML
# ----------------------------------------------------------------------------
def esc(x) -> str:
    return html.escape(str(x))


def pct(p: float, decimals: int | None = None) -> str:
    """0.985 -> '98.5%'. Whole numbers in the middle, one decimal near 0 and 100."""
    v = 100.0 * float(p)
    if decimals is None:
        if v < 0.1:
            return "under 0.1%"
        if v > 99.9:
            return "over 99.9%"
        decimals = 1 if (v < 10 or v > 90) else 0
    return f"{v:.{decimals}f}%"


def pct_pair(p: float, threshold: float) -> tuple[str, str]:
    """Probability and threshold as text, with enough decimals to tell them apart."""
    a, b = pct(p), pct(threshold)
    for decimals in (1, 2, 3, 4):
        if a != b or p == threshold:
            break
        a, b = pct(p, decimals), pct(threshold, decimals)
    return a, b


def train_groups(cfg: dict) -> set | None:
    """Patient groups a model was trained on: {'children'}, {'children', 'adults'}, or None if not recorded."""
    groups = cfg.get("train_groups")
    if isinstance(groups, (list, tuple)):
        return {str(g).strip().lower() for g in groups}
    described = " ".join(str(s) for s in (cfg.get("train_data") or [])).lower()   # older configs
    found = {g for g, word in (("children", "child"), ("adults", "adult")) if word in described}
    return found or None


def join_words(items) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def trained_on_text(cfg: dict) -> str:
    data = [str(s) for s in (cfg.get("train_data") or [])]
    return "Trained on " + " + ".join(data) + "." if data else "Training data: not recorded."


def familiarity_html(r: dict, predictor: Predictor) -> tuple[str, bool]:
    """-> (HTML, True if the result should not be relied on)."""
    level = r["ood_level"]
    if level is None:
        return ('<div class="ps-check"><b>X-ray check:</b> not available for this model yet, so it cannot '
                'warn you about unusual X-rays.</div>'), False
    if level == "ok":
        return ('<div class="ps-check ps-check--ok"><b>X-ray check passed:</b> this X-ray looks like the '
                'X-rays this model learned from.</div>'), False
    p, n = r["ood_percentile"], len(predictor.ood.ref_scores)
    if p >= 100:
        how = f"None of the {n} comparison X-rays looked this unfamiliar to the model."
    elif p >= 99.5:
        how = "More than 99 in 100 comparison X-rays looked more familiar to the model than this one."
    else:
        how = f"About {p:.0f} in 100 comparison X-rays looked more familiar to the model than this one."
    if level == "very unusual":
        head, blocking = "This X-ray looks unlike the X-rays this model learned from. Do not rely on this result.", True
    else:
        head, blocking = ("This X-ray looks somewhat unlike the X-rays this model learned from. "
                          "Treat this result with caution."), False
    return f'<div class="ps-alert" role="alert"><b>{head}</b><span>{how}</span></div>', blocking


def patient_group_html(cfg: dict, age: str) -> tuple[str, bool]:
    """-> (HTML, True if the result should not be relied on)."""
    if age not in AGE_TO_GROUP:
        return ('<div class="ps-check"><b>Patient check:</b> not done. Choose the patient age group '
                'to check whether this model was trained on such patients.</div>'), False
    groups = train_groups(cfg)
    if groups is None:
        return ('<div class="ps-check"><b>Patient check:</b> not possible, the patient groups this model '
                'was trained on are not recorded.</div>'), False
    if AGE_TO_GROUP[age] in groups:
        return (f'<div class="ps-check ps-check--ok"><b>Patient check passed:</b> this model was trained on '
                f'X-rays of {AGE_WORDS[age]}.</div>'), False
    known = join_words(GROUP_WORDS.get(g, g) for g in sorted(groups, key=lambda g: (g != "children", g)))
    return (f'<div class="ps-alert" role="alert"><b>This model was never trained on X-rays of {AGE_WORDS[age]}. '
            f'Its result for this patient may be wrong, even when it looks confident.</b>'
            f'<span>It only learned from X-rays of {esc(known)}.</span></div>'), True


def known_answer_html(example: dict | None, r: dict) -> str:
    """For our examples: compare the model with the answer recorded in the dataset."""
    if not example:
        return ""
    truth = esc(example.get("true_label", ""))
    label = example.get("label")
    if label in (0, 1):
        right = bool(r["flagged"]) == bool(label)
        if right:
            text = f"<b>Correct.</b> The known answer for this example is: {truth}."
        elif label == 1:
            text = f"<b>Missed.</b> The known answer for this example is: {truth}."
        else:
            text = f"<b>False alarm.</b> The known answer for this example is: {truth}."
    else:                              # a picture that is not an X-ray: the right reaction is a warning
        if r["ood_level"] is None:
            right, text = False, "<b>No warning.</b> This picture is not an X-ray, and this model has no X-ray check yet."
        elif r["ood_level"] == "ok":
            right, text = False, "<b>Missed.</b> This picture is not an X-ray, but the X-ray check did not notice."
        else:
            right, text = True, "<b>Correct warning.</b> This picture is not an X-ray."
    return f'<div class="ps-truth {"ps-truth--right" if right else "ps-truth--wrong"}">{text}</div>'


def status_html(model: dict, r: dict | None, age: str, example: dict | None) -> str:
    """Everything one model says about one picture."""
    if r is None:
        return '<div class="ps-empty">This model could not analyse the picture.</div>'
    p, thr = r["probability"], r["threshold"]
    familiarity, block_a = familiarity_html(r, model["predictor"])
    patient, block_b = patient_group_html(model["config"], age)
    p_text, thr_text = pct_pair(p, thr)

    if r["flagged"]:
        kind, title = "flag", "Flag for priority review"
        detail = f"this model flags at {thr_text} or above"
    elif p < 0.5:
        kind, title = "clear", "No sign of pneumonia found"
        detail = f"this model flags at {thr_text} or above"
    else:       # only possible with a threshold above 50%: saying "no sign" would contradict the number
        kind, title = "unsure", "Not flagged, but not clearly normal"
        detail = f"high, but below the {thr_text} at which this model flags"
    if block_a or block_b:
        kind = "muted"
    verdict = (
        f'<div class="ps-verdict ps-verdict--{kind}">'
        f'<div class="ps-verdict__title">{title}</div>'
        f'<div class="ps-verdict__sub">Estimated chance of pneumonia: <b>{p_text}</b> <span>({detail})</span></div>'
        f'<div class="ps-bar" aria-hidden="true"><div class="ps-bar__fill" style="width:{100 * p:.1f}%"></div>'
        f'<div class="ps-bar__tick" style="left:{100 * min(max(thr, 0.0), 1.0):.1f}%"></div></div>'
        f'</div>')
    alerts = "".join(h for h in (familiarity, patient) if "ps-alert" in h)
    checks = "".join(h for h in (familiarity, patient) if "ps-alert" not in h)
    return f'<div class="ps-status">{verdict}{alerts}{checks}{known_answer_html(example, r)}</div>'


EMPTY_STATUS = '<div class="ps-empty">Upload a chest X-ray or pick an example to see what this model says.</div>'


def _metric(value, kind: str) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if v != v:                         # NaN
        return "n/a"
    if kind == "count":
        return f"{int(v):,}"
    return f"{100 * v:.0f}%" if kind == "share" else f"{v:.3f}"


def _interval(ci, kind: str) -> str:
    try:
        lo, hi = float(ci[0]), float(ci[1])
        if lo != lo or hi != hi:
            return ""
    except (TypeError, ValueError, IndexError):
        return ""
    text = f"{100 * lo:.0f} to {100 * hi:.0f}%" if kind == "share" else f"{lo:.3f} to {hi:.3f}"
    return f'<small title="95% confidence interval">{text}</small>'


def tested_html(cfg: dict) -> str:
    """The "How well was this model tested?" panel. Shows only numbers stored in the model's config."""
    head = '<div class="ps-tested__title">How well was this model tested?</div>'
    metrics = cfg.get("metrics")
    rows = []
    if isinstance(metrics, dict):
        for group, m in metrics.items():
            if not isinstance(m, dict):
                continue
            name = str(group).replace("_", " ").strip()
            cells = [f"<td>{_metric(m.get('n'), 'count')}</td>"]
            for key, kind in (("sensitivity", "share"), ("specificity", "share"), ("auroc", "score")):
                cells.append(f"<td>{_metric(m.get(key), kind)}{_interval(m.get(key + '_ci'), kind)}</td>")
            rows.append(f"<tr><th scope='row'>{esc(name[:1].upper() + name[1:])}</th>{''.join(cells)}</tr>")
    if not rows:
        return f'<div class="ps-tested">{head}<div class="ps-tested__none">Not measured yet.</div></div>'
    return (
        f'<div class="ps-tested">{head}<div class="ps-tablewrap"><table class="ps-table"><thead><tr>'
        '<th scope="col">Test X-rays (never used for training)</th><th scope="col">Number</th>'
        '<th scope="col">Sensitivity</th><th scope="col">Specificity</th><th scope="col">AUROC</th>'
        f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
        '<div class="ps-note">Sensitivity: the share of X-rays with pneumonia that the model flagged. '
        'Specificity: the share of X-rays without pneumonia that it did not flag. '
        'AUROC: how well it ranks sick above healthy (0.5 is guessing, 1 is perfect). '
        'Small print: the range we are 95% sure the true value lies in.</div></div>')


def notice_html(text: str = "", kind: str = "info") -> str:
    return f'<div class="ps-notice ps-notice--{kind}" role="status">{text}</div>' if text else ""


def example_notice(example: dict) -> str:
    age = f", age {int(example['age'])}" if isinstance(example.get("age"), (int, float)) else ""
    return notice_html(
        f"<b>Example with a known answer: {esc(example.get('true_label', ''))}{age}.</b> "
        f"<span>{esc(example.get('source', ''))}</span>")


# ----------------------------------------------------------------------------
# Event handlers. They never raise: every problem becomes a message on the page.
# ----------------------------------------------------------------------------
def render(path: str | None, age: str | None) -> list:
    """-> values for RESULT_OUTPUTS: the notice, then per model (status, picture row, input picture, heatmap)."""
    age = age if age in dict((v, k) for k, v in AGE_CHOICES) else "unknown"

    def blank(notice=""):
        return [notice] + [x for _ in MODELS for x in (EMPTY_STATUS, gr.update(visible=False), None, None)]

    if not path:
        return blank()
    try:
        entry = run_models(str(path))
    except ValueError as e:
        return blank(notice_html(f"<b>This file cannot be used.</b> <span>{esc(e)} Please upload a chest X-ray "
                                 f"saved as a PNG or JPEG picture.</span>", "error"))
    except Exception:
        log(f"unexpected error:\n{traceback.format_exc()}")
        return blank(notice_html("<b>Something went wrong while analysing this picture.</b> "
                                 "<span>Please try again or use another picture.</span>", "error"))

    example = entry["example"]
    out = [example_notice(example) if example else ""]
    for model, r in zip(MODELS, entry["results"]):
        out.append(status_html(model, r, age, example))
        if r is None:
            out += [gr.update(visible=False), None, None]
        else:
            out += [gr.update(visible=True), r["input_image"], r["heatmap"]]
    return out


def on_upload(path):
    """A new picture is a new patient: the age group goes back to 'Not sure'."""
    values = render(path, "unknown")
    unreadable = bool(path) and "ps-notice--error" in str(values[0])
    return [gr.update(value=None) if unreadable else gr.update(), "unknown", gr.update(selected_index=None)] + values


def on_clear():
    return ["unknown", gr.update(selected_index=None)] + render(None, "unknown")


def on_age(path, age):
    return render(path, age)


def on_example(evt: gr.SelectData):
    try:
        example = EXAMPLES[int(evt.index)]
    except (TypeError, ValueError, IndexError):
        return [gr.update(), gr.update()] + render(None, "unknown")
    age = {"children": "child_1_5", "adults": "adult"}.get(example.get("group"), "unknown")
    return [example["path"], age] + render(example["path"], age)


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
class XrayUpload(gr.Image):
    """gr.Image that passes the uploaded file path to our code untouched.

    The stock component opens and converts the file before our code runs. That would
    flatten 16-bit X-rays to white and turn an unreadable file into a generic error.
    Here Predictor reads the file itself and we word the message."""

    is_template = True

    def preprocess(self, payload):
        return str(payload.path) if payload is not None and payload.path else None


SYSTEM_FONT = ["ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "Roboto", "Helvetica Neue", "Arial", "sans-serif"]
THEME = gr.themes.Default(      # system fonts only: nothing is fetched from the internet during a demo
    primary_hue="blue", neutral_hue="slate", radius_size="md", spacing_size="sm",
    font=SYSTEM_FONT, font_mono=["ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace"])

CSS = """
.gradio-container { max-width: 1360px !important; margin: 0 auto !important;
  --ps-flag: #b42318; --ps-flag-soft: #fef3f2; --ps-clear: #067647; --ps-clear-soft: #ecfdf3;
  --ps-warn: #7a2e0e; --ps-warn-soft: #fffaeb; --ps-warn-line: #f79009; --ps-soft: var(--background-fill-secondary); }
.dark .gradio-container, .gradio-container.dark {
  --ps-flag: #fda29b; --ps-flag-soft: rgba(240, 68, 56, .16); --ps-clear: #75e0a7; --ps-clear-soft: rgba(23, 178, 106, .14);
  --ps-warn: #fec84b; --ps-warn-soft: rgba(247, 144, 9, .14); --ps-warn-line: #f79009; }
.ps-header { display: flex; flex-wrap: wrap; align-items: flex-start; justify-content: space-between; gap: 8px 24px; }
.ps-header h1 { font-size: 1.45rem; font-weight: 700; margin: 0; line-height: 1.25; }
.ps-header h1 span { font-weight: 400; color: var(--body-text-color-subdued); }
.ps-header p { margin: 4px 0 0; max-width: 62rem; line-height: 1.4; }
.ps-badge { border: 1px solid var(--ps-warn-line); background: var(--ps-warn-soft); color: var(--ps-warn);
  border-radius: 999px; padding: 4px 12px; font-size: .82rem; font-weight: 600; white-space: nowrap; }
.ps-model h2 { font-size: 1.05rem; font-weight: 700; margin: 0; }
.ps-model p { margin: 2px 0 0; font-size: .82rem; color: var(--body-text-color-subdued); }
.ps-status { display: flex; flex-direction: column; gap: 6px; }
.ps-empty { padding: 14px; border: 1px dashed var(--border-color-primary); border-radius: var(--radius-md);
  color: var(--body-text-color-subdued); font-size: .92rem; }
.ps-verdict { border-left: 5px solid var(--border-color-primary); background: var(--ps-soft);
  border-radius: var(--radius-md); padding: 8px 12px 10px; }
.ps-verdict__title { font-size: 1.2rem; font-weight: 700; line-height: 1.3; }
.ps-verdict__sub { font-size: .9rem; margin-top: 2px; }
.ps-verdict__sub span { color: var(--body-text-color-subdued); }
.ps-verdict--flag { border-left-color: var(--ps-flag); background: var(--ps-flag-soft); }
.ps-verdict--flag .ps-verdict__title { color: var(--ps-flag); }
.ps-verdict--clear { border-left-color: var(--ps-clear); background: var(--ps-clear-soft); }
.ps-verdict--clear .ps-verdict__title { color: var(--ps-clear); }
.ps-verdict--muted .ps-verdict__title { color: var(--body-text-color-subdued); }
.ps-bar { position: relative; height: 6px; margin-top: 8px; border-radius: 3px; background: var(--border-color-primary); }
.ps-bar__fill { height: 100%; border-radius: 3px; background: var(--body-text-color-subdued); }
.ps-verdict--flag .ps-bar__fill { background: var(--ps-flag); }
.ps-verdict--clear .ps-bar__fill { background: var(--ps-clear); }
.ps-bar__tick { position: absolute; top: -4px; width: 2px; height: 14px; margin-left: -1px; background: var(--body-text-color); }
.ps-alert { border: 1px solid var(--ps-warn-line); border-left-width: 5px; background: var(--ps-warn-soft);
  color: var(--ps-warn); border-radius: var(--radius-md); padding: 8px 12px; font-size: .92rem; line-height: 1.35; }
.ps-alert b { display: block; }
.ps-alert span { display: block; margin-top: 2px; font-size: .84rem; }
.ps-check, .ps-truth { font-size: .84rem; line-height: 1.35; color: var(--body-text-color-subdued); padding: 0 2px; }
.ps-check--ok b { color: var(--ps-clear); }
.ps-truth { color: var(--body-text-color); border-top: 1px solid var(--border-color-primary); padding-top: 6px; }
.ps-truth--right b { color: var(--ps-clear); }
.ps-truth--wrong b { color: var(--ps-flag); }
.ps-caption { font-size: .8rem; color: var(--body-text-color-subdued); line-height: 1.35; }
.ps-tested__title { font-size: .92rem; font-weight: 700; }
.ps-tested__none { font-size: .88rem; color: var(--body-text-color-subdued); }
.ps-tablewrap { overflow-x: auto; }
.ps-table { width: 100%; border-collapse: collapse; font-size: .82rem; margin-top: 4px; }
.ps-table th, .ps-table td { text-align: right; padding: 3px 6px; border-bottom: 1px solid var(--border-color-primary);
  vertical-align: top; font-weight: 400; }
.ps-table thead th { font-weight: 600; color: var(--body-text-color-subdued); }
.ps-table th:first-child { text-align: left; padding-left: 0; }
.ps-table td { font-variant-numeric: tabular-nums; font-weight: 600; white-space: nowrap; }
.ps-table small { display: block; font-weight: 400; color: var(--body-text-color-subdued); }
.ps-note { font-size: .76rem; color: var(--body-text-color-subdued); line-height: 1.35; margin-top: 4px; }
.ps-notice { border: 1px solid var(--border-color-primary); background: var(--ps-soft); border-radius: var(--radius-md);
  padding: 8px 12px; font-size: .9rem; line-height: 1.35; }
.ps-notice span { color: var(--body-text-color-subdued); }
.ps-notice--error { border-color: var(--ps-flag); background: var(--ps-flag-soft); }
.ps-limits { font-size: .86rem; line-height: 1.45; }
.ps-limits h2 { font-size: .95rem; font-weight: 700; margin: 0 0 4px; }
.ps-limits ul { margin: 0; padding-left: 1.2rem; }
.ps-limits p { margin: 6px 0 0; color: var(--body-text-color-subdued); }
"""

HEADER = """
<div class="ps-header">
  <div>
    <h1>PneumoShift <span>&nbsp;a second look at a chest X-ray for pneumonia</span></h1>
    <p>For a clinic without a radiologist. Upload a chest X-ray to see whether a doctor should review it first.
    Each AI model shows where it looked, and warns you when the X-ray or the patient is unlike what it learned from.</p>
  </div>
  <div class="ps-badge">Research demo. Not a medical device. Not for diagnosis.</div>
</div>
"""

CAPTION = ('<div class="ps-caption">Left: the picture as the model received it (grey, cropped to a square, 224 pixels). '
           'Right: bright colours mark the areas that pushed the model towards pneumonia. The colours are scaled per '
           'picture, so there is always a brightest spot, even when the model finds little.</div>')


def limits_html() -> str:
    sources = []
    for e in EXAMPLES:
        if e.get("source") and e["source"] not in sources and e.get("label") in (0, 1):
            sources.append(e["source"])
    credit = f"<p>Example X-rays: {esc(' '.join(sources))}</p>" if sources else ""
    return f"""
<div class="ps-limits">
  <h2>Limits you should know about</h2>
  <ul>
    <li><b>Research demo. Not a medical device. Not for diagnosis.</b> A student project that no authority has
        checked. It can only suggest which X-rays to look at first. The decision stays with a clinician.</li>
    <li>A model can be very accurate on its own test X-rays and still be wrong for a patient unlike the ones it
        learned from. The two checks catch many such cases, not all. No warning does not mean the result is right.</li>
    <li>The models learned from public X-rays of two groups only: children aged 1 to 5 from one hospital in
        Guangzhou, China, and (second model) adults from one hospital in the USA. Babies, older children,
        teenagers, other hospitals and other X-ray machines are untested.</li>
    <li>The models know two answers, pneumonia or not. They do not recognise other problems on an X-ray.
        The heatmap shows what influenced the model. It is not a map of the disease.</li>
    <li>Please do not upload pictures that show a patient's name or other personal details.</li>
  </ul>
  {credit}
</div>"""


def build() -> gr.Blocks:
    with gr.Blocks(title="PneumoShift: chest X-ray second look", analytics_enabled=False) as demo:
        gr.HTML(HEADER)
        if not MODELS:
            gr.HTML(notice_html(
                "<b>No model could be loaded, so the demo cannot run.</b> <span>If this is your demo: check that the "
                "model repositories exist and are public, then restart. The start-up log names the reason.</span>", "error"))
            gr.HTML(limits_html())
            return demo

        with gr.Row():
            with gr.Column(scale=4, min_width=300):
                image = XrayUpload(
                    label="Chest X-ray", type="filepath", image_mode=None, sources=["upload"], height=250,
                    buttons=[], placeholder="Drop a chest X-ray here\nor click to choose a file (PNG or JPEG)")
                age = gr.Radio(
                    choices=AGE_CHOICES, value="unknown", label="Patient age group (optional)",
                    info="Lets the tool check whether each model was trained on patients like this one.")
                gallery = None
                if EXAMPLES:
                    gallery = gr.Gallery(
                        value=[(e["path"], str(e.get("caption") or e.get("true_label") or "")) for e in EXAMPLES],
                        label="Examples with a known answer (click one)", columns=4, height="auto",
                        allow_preview=False, object_fit="cover", interactive=False, buttons=[], format="png",
                        elem_id="ps-examples")

            with gr.Column(scale=9, min_width=300):
                notice = gr.HTML("", elem_id="ps-notice")
                result_outputs = [notice]
                with gr.Row():
                    for i, m in enumerate(MODELS):
                        with gr.Column(variant="panel", min_width=300, elem_classes="ps-card"):
                            gr.HTML(f'<div class="ps-model"><h2>{esc(m["name"])}</h2>'
                                    f'<p>{esc(trained_on_text(m["config"]))}</p></div>')
                            status = gr.HTML(EMPTY_STATUS, elem_id=f"ps-status-{i}")
                            with gr.Row(visible=False) as pictures:
                                saw = gr.Image(label="What the model saw", interactive=False, format="png",
                                               buttons=["fullscreen"], min_width=120, elem_id=f"ps-saw-{i}")
                                heat = gr.Image(label="Where the model looked", interactive=False, format="png",
                                                buttons=["fullscreen"], min_width=120, elem_id=f"ps-heat-{i}")
                            gr.HTML(tested_html(m["config"]))
                            result_outputs += [status, pictures, saw, heat]
                gr.HTML(CAPTION)

        gr.HTML(limits_html())

        quiet = dict(show_progress="hidden", api_visibility="private")
        deselect = [gallery] if gallery is not None else []
        image.upload(on_upload if gallery is not None else (lambda p: on_upload(p)[:2] + on_upload(p)[3:]),
                     inputs=image, outputs=[image, age] + deselect + result_outputs, **quiet)
        image.clear(on_clear if gallery is not None else (lambda: on_clear()[:1] + on_clear()[2:]),
                    outputs=[age] + deselect + result_outputs, **quiet)
        age.input(on_age, inputs=[image, age], outputs=result_outputs, **quiet)
        if gallery is not None:
            gallery.select(on_example, outputs=[image, age] + result_outputs, **quiet)
    return demo


demo = build()

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2, max_size=40)
    demo.launch(theme=THEME, css=CSS, show_error=False, ssr_mode=False, max_file_size=MAX_UPLOAD,
                footer_links=["gradio", "settings"])
