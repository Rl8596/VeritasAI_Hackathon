"""Step 12: a picture of the network with nodes and weights (for the write-up and the video).

    python scripts/12_network_figure.py            # writes docs/network_nodes.svg

Nothing in the picture is invented: every line is a weight of the published model `kids_plus_adults`, and the
brightness of every node is how strongly that part of the network responds to the example X-ray. The network has
11 million weights, so the picture shows a sample: 10 to 16 feature maps per layer, and between two stages the
first layer of the next stage (one line = one 3 x 3 filter). The last layer is drawn completely (512 weights).
"""
import argparse
import base64
import io
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pneumoshift import modeling, preprocess  # noqa: E402

W, H, MID = 1500, 835, 410
BLUE, ORANGE = "#38bdf8", "#fb923c"
COLS = [("First layer", 310, 10), ("Stage 1", 470, 10), ("Stage 2", 630, 12), ("Stage 3", 790, 14), ("Stage 4", 950, 16)]
STRIP_X, STRIP_H, OUT_X = 1110, 560, 1385


def node_ys(k):
    step = min(46, 560 / (k - 1))
    return [MID + (i - (k - 1) / 2) * step for i in range(k)]


def curve(x1, y1, x2, y2):
    xm = (x1 + x2) / 2
    return f"M{x1:.1f} {y1:.1f} C{xm:.1f} {y1:.1f} {xm:.1f} {y2:.1f} {x2:.1f} {y2:.1f}"


def lines(x1, ys1, x2, ys2, weight):
    """weight[j, i]: signed strength of the connection from node i (left) to node j (right)."""
    s = np.abs(weight) / np.abs(weight).max()
    out = []
    for j, i in sorted(np.ndindex(*weight.shape), key=lambda ji: s[ji]):          # strongest on top
        v = s[j, i] ** 1.5
        out.append(f'<path d="{curve(x1, ys1[i], x2, ys2[j])}" stroke="{BLUE if weight[j, i] > 0 else ORANGE}" '
                   f'stroke-width="{0.3 + 2.0 * v:.2f}" stroke-opacity="{0.05 + 0.85 * v:.2f}"/>')
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/kids_plus_adults", help="folder or Hub repo of a trained model")
    ap.add_argument("--example", default="web/examples/child_pneumonia_1.png")
    ap.add_argument("--out", default="docs/network_nodes.svg")
    args = ap.parse_args()

    model, cfg = modeling.load_checkpoint(args.model)
    x, gray = preprocess.pil_to_model_input(Image.open(ROOT / args.example))
    with torch.no_grad():
        a = [model.act1(model.bn1(model.conv1(x)))]
        a.append(model.layer1(model.maxpool(a[0])))
        for layer in (model.layer2, model.layer3, model.layer4):
            a.append(layer(a[-1]))
        emb = a[-1].mean((2, 3))[0]
        logit = float(model.fc(emb[None])[0, 0])
    prob = float(modeling.sigmoid(np.array([logit]), cfg["temperature"])[0])
    flagged = prob >= cfg["threshold"]
    acts = [t.mean((2, 3))[0].numpy() for t in a]
    sd = model.state_dict()
    count = lambda *prefixes: sum(p.numel() for n, p in model.named_parameters() if n.startswith(prefixes))
    n_weights = [count("conv1", "bn1")] + [count(f"layer{i}") for i in (1, 2, 3, 4)]
    total = sum(p.numel() for p in model.parameters())
    assert sum(n_weights) + count("fc") == total

    # the connections that are drawn: picture -> first layer, then the first layer of each next stage
    kernels = [sd["conv1.weight"].sum(1, keepdim=True)] + [sd[f"layer{i}.0.conv1.weight"] for i in (1, 2, 3, 4)]
    shown = [np.linspace(0, len(v) - 1, k).round().astype(int) for v, (_, _, k) in zip(acts, COLS)]
    grids = ["56 × 56", "56 × 56", "28 × 28", "14 × 14", "7 × 7"]

    g_lines, g_nodes, g_text = [], [], []
    img_x, img_s = 30, 170
    prev_x, prev_ys, prev_idx = img_x + img_s + 4, None, np.array([0])
    for c, ((name, cx, k), idx) in enumerate(zip(COLS, shown)):
        ys = node_ys(k)
        ker = kernels[c][idx][:, prev_idx].numpy()                                 # [right nodes, left nodes, 3, 3]
        strength = np.sqrt((ker ** 2).sum((2, 3))) * np.sign(ker.sum((2, 3)))
        if prev_ys is None:
            prev_ys = [MID] * 1
            g_lines += lines(prev_x, [MID - img_s / 2 + img_s * (j + 0.5) / k for j in range(k)], cx - 10, ys,
                             np.diag(strength[:, 0]))
        else:
            g_lines += lines(prev_x + 10, prev_ys, cx - 10, ys, strength)
        bright = acts[c][idx] / acts[c][idx].max()
        for y, b in zip(ys, bright):
            g_nodes.append(f'<circle cx="{cx}" cy="{y:.1f}" r="9" fill="#a5f3fc" fill-opacity="{0.12 + 0.88 * b:.2f}" '
                           f'stroke="#67e8f9" stroke-opacity="0.8" stroke-width="1.2"/>')
        n_maps = len(acts[c])
        for row, (txt, cls) in enumerate([(name, "h"), (f"{n_maps} maps, {grids[c]}", "p"),
                                          (f"{n_weights[c]:,} weights", "p"), (f"{k} of {n_maps} shown", "d")]):
            g_text.append(f'<text class="{cls}" x="{cx}" y="{722 + 18 * row}">{txt}</text>')
        prev_x, prev_ys, prev_idx = cx, ys, idx

    # the 512 numbers (average of each map of stage 4) and the last layer, drawn completely
    top = MID - STRIP_H / 2
    strip_y = lambda i: top + STRIP_H * (i + 0.5) / 512
    e = emb.numpy() / emb.numpy().max()
    for y, i in zip(prev_ys, prev_idx):
        g_lines.append(f'<path d="{curve(prev_x + 10, y, STRIP_X - 2, strip_y(i))}" stroke="#94a3b8" stroke-opacity="0.45" '
                       f'stroke-width="0.8" stroke-dasharray="3 3"/>')
    for i in range(512):
        g_nodes.append(f'<rect x="{STRIP_X}" y="{strip_y(i) - 0.55:.2f}" width="16" height="1.1" fill="#a5f3fc" '
                       f'fill-opacity="{0.1 + 0.9 * e[i]:.2f}"/>')
    fc = sd["fc.weight"][0].numpy()
    g_lines += lines(STRIP_X + 18, [strip_y(i) for i in range(512)], OUT_X - 36, [MID], fc[None, :])
    for row, (txt, cls) in enumerate([("512 numbers", "h"), ("average of each map", "p"), ("no weights", "p")]):
        g_text.append(f'<text class="{cls}" x="{STRIP_X + 8}" y="{722 + 18 * row}">{txt}</text>')
    for row, (txt, cls) in enumerate([("Last layer", "h"), ("512 numbers in, 1 out", "p"), (f"{count('fc')} weights", "p"),
                                      ("all 512 lines drawn", "d")]):
        g_text.append(f'<text class="{cls}" x="{OUT_X}" y="{722 + 18 * row}">{txt}</text>')
    colour = "#fb7185" if flagged else "#4ade80"
    out_node = (f'<circle cx="{OUT_X}" cy="{MID}" r="34" fill="{colour}" fill-opacity="0.9" stroke="#ffffff" stroke-opacity="0.7" '
                f'stroke-width="1.5" filter="url(#glow)"/>'
                f'<text x="{OUT_X}" y="{MID + 7}" text-anchor="middle" font-size="18" font-weight="700" fill="#0b1020">{prob:.1%}</text>'
                f'<text class="h" x="{OUT_X}" y="{MID + 62}">pneumonia score</text>'
                f'<text class="p" x="{OUT_X}" y="{MID + 80}">threshold {cfg["threshold"]:.0%}:</text>'
                f'<text class="p" x="{OUT_X}" y="{MID + 98}" fill="{colour}">{"flag for review" if flagged else "no flag"}</text>')

    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(gray)).save(buf, format="PNG", optimize=True)
    picture = base64.b64encode(buf.getvalue()).decode()
    iy = MID - img_s / 2
    for row, (txt, cls) in enumerate([("Chest X-ray", "h"), ("224 × 224 pixels", "p"), ("as the model sees it", "d")]):
        g_text.append(f'<text class="{cls}" x="{img_x + img_s / 2}" y="{722 + 18 * row}">{txt}</text>')

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img"
     aria-label="The neural network of PneumoShift with nodes and weights">
<style>
 text {{ font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }}
 .h {{ font-size: 14px; font-weight: 700; fill: #e2e8f0; text-anchor: middle; }}
 .p {{ font-size: 13px; fill: #cbd5e1; text-anchor: middle; }}
 .d {{ font-size: 12px; fill: #7c8aa0; text-anchor: middle; }}
</style>
<defs>
 <radialGradient id="bg" cx="50%" cy="45%" r="75%"><stop offset="0" stop-color="#16213a"/><stop offset="1" stop-color="#070b16"/></radialGradient>
 <filter id="glow" x="-80%" y="-80%" width="260%" height="260%"><feGaussianBlur stdDeviation="4" result="b"/>
  <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
 <clipPath id="clip"><rect x="{img_x}" y="{iy}" width="{img_s}" height="{img_s}" rx="10"/></clipPath>
</defs>
<rect width="{W}" height="{H}" fill="url(#bg)"/>
<text x="30" y="46" font-size="24" font-weight="700" fill="#f8fafc">PneumoShift's neural network: ResNet-18, {total:,} weights</text>
<text x="30" y="74" font-size="14" fill="#cbd5e1">Every line is a real weight of our trained model: <tspan fill="{BLUE}" font-weight="700">blue = positive</tspan>, <tspan fill="{ORANGE}" font-weight="700">orange = negative</tspan>, thicker = stronger. Node brightness = how strongly that part of the network responds to this X-ray.</text>
<g fill="none" stroke-linecap="round">{"".join(g_lines)}</g>
<image href="data:image/png;base64,{picture}" x="{img_x}" y="{iy}" width="{img_s}" height="{img_s}" clip-path="url(#clip)"/>
<rect x="{img_x}" y="{iy}" width="{img_s}" height="{img_s}" rx="10" fill="none" stroke="#67e8f9" stroke-opacity="0.6" stroke-width="1.5"/>
<g filter="url(#glow)">{"".join(g_nodes)}</g>
{out_node}
{"".join(g_text)}
<text x="30" y="{H - 14}" font-size="12" fill="#7c8aa0">Simplified: a sample of the feature maps is shown, and between two stages one line stands for one 3 × 3 filter (9 weights) of the next stage's first layer. The last layer is complete.</text>
</svg>
'''
    out = ROOT / args.out
    out.write_text(svg)
    print(f"wrote {out} ({len(svg) / 1024:.0f} KB): score {prob:.3f}, threshold {cfg['threshold']:.3f}, flagged {flagged}, "
          f"{total:,} weights, per part {n_weights} + {count('fc')}")
