"""One look for every figure: same colours for the same groups, quiet axes, readable on a slide."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Colour follows the group, in every figure.
CHILDREN = "#2a78d6"   # blue   = children (Kermany, Guangzhou)
ADULTS = "#eb6834"     # orange = adults (RSNA, USA)
THIRD = "#1baf7a"      # aqua   = a third series when needed (e.g. "our fix")
NEUTRAL = "#8a8983"    # grey   = baselines, chance level, reference lines
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e6e5e0"


def use_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 12, "axes.labelcolor": TEXT_2, "text.color": TEXT,
        "xtick.color": TEXT_2, "ytick.color": TEXT_2, "xtick.labelsize": 11, "ytick.labelsize": 11,
        "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "lines.linewidth": 2, "lines.markersize": 7, "legend.frameon": False, "legend.fontsize": 11,
    })


def finish(fig, path, note: str | None = None) -> None:
    """Save and close. `note` is a small grey footnote (data source, sample size)."""
    if note:
        fig.text(0.01, -0.02, note, ha="left", va="top", fontsize=9, color=TEXT_2)
    fig.savefig(path)
    plt.close(fig)
