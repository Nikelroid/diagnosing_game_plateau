"""One place where every figure's colour, type and spacing decisions live.

Palette provenance: these are slots 1 to 7 of the reference categorical palette, in its fixed
documented order, which is validated for colourblind separation on the adjacent pairlist. Families
are assigned to slots once, here, and never cycled, so a family keeps its colour across every
figure and across reruns that drop or add games.

Colour is never the only channel. Each family also owns a marker shape, because a scatter puts all
seven families on one panel where the adjacent-pair guarantee does not apply, and because these
figures have to survive a greyscale print.
"""

from __future__ import annotations

import matplotlib

# --- typography -------------------------------------------------------------------------------
# DejaVu is what matplotlib ships and it looks like software, not like a paper. TeX Live provides
# TeX Gyre Heros, a Helvetica-class face that sits properly beside the Times body text a NeurIPS or
# AAAI template sets. We register the OTFs directly rather than relying on a system font cache.
import glob
from matplotlib import font_manager

_TEXGYRE = "/apps/generic/texlive/2026/texmf-dist/fonts/opentype/public/tex-gyre"


def _register_fonts():
    found = []
    for pattern in ("texgyreheros-*.otf", "texgyretermes-*.otf"):
        for path in glob.glob(f"{_TEXGYRE}/{pattern}"):
            try:
                font_manager.fontManager.addfont(path)
                found.append(path)
            except Exception:  # noqa: BLE001 - fall back to DejaVu rather than fail a figure
                pass
    names = {f.name for f in font_manager.fontManager.ttflist}
    return "TeX Gyre Heros" if "TeX Gyre Heros" in names else "DejaVu Sans"


SANS = _register_fonts()


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

__all__ = ["FAMILY_COLOR", "FAMILY_MARKER", "FAMILY_LABEL", "INK", "MUTED", "RULE", "apply_style",
           "finish"]

# fixed slot order: blue, orange, aqua, yellow, magenta, green, violet
FAMILY_COLOR = {
    "card": "#2a78d6",
    "dial": "#eb6834",
    "board": "#1baf7a",
    "dice": "#eda100",
    "comm": "#e87ba4",
    "multi": "#008300",
    "solo": "#4a3aa7",
}
FAMILY_MARKER = {
    "card": "o",
    "dial": "D",
    "board": "s",
    "dice": "^",
    "comm": "v",
    "multi": "P",
    "solo": "X",
}
FAMILY_LABEL = {
    "card": "Card, 2p",
    "dial": "Gin Rummy dial",
    "board": "Board and fog of war",
    "dice": "Dice",
    "comm": "Bargaining, signalling",
    "multi": "Card, 3-4p",
    "solo": "Solo vs chance",
}

INK = "#1a1a19"
MUTED = "#6b6a63"
RULE = "#d8d6cc"
ANCHOR = "#9a6b12"


def apply_style():
    """Recessive axes, quiet grid, one type family. Called by every figure script."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": SANS,
            "font.size": 9.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlepad": 12,
            "axes.titlelocation": "left",
            "axes.labelsize": 9,
            "axes.labelcolor": INK,
            "axes.edgecolor": RULE,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "grid.color": RULE,
            "grid.linewidth": 0.55,
            "grid.linestyle": (0, (1, 3)),
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.axisbelow": True,
            "font.stretch": "normal",
            "legend.frameon": False,
            "legend.fontsize": 8,
            "text.color": INK,
        }
    )


def finish(fig, path_stem: str):
    """Save the same figure as PDF for the paper and PNG for quick eyeballing."""
    fig.savefig(f"{path_stem}.pdf")
    fig.savefig(f"{path_stem}.png")
    plt.close(fig)
    print(f"  wrote {path_stem}.pdf and .png")
