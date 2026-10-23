"""Draw the GlucoTwin architecture diagram.

Outputs
-------
docs/architecture.png   300 dpi raster for the README and slides
docs/architecture.pdf   vector copy for printing

Run:  python src/make_architecture.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

INK = "#1c1b19"
MUTED = "#6f6a60"
PAPER = "#f7f4ed"
LINE = "#c9c2b4"
CLAY = "#b4552d"
SAGE = "#6f8060"
OCHRE = "#b08a4f"
SLATE = "#4f6b7a"

plt.rcParams["font.family"] = "DejaVu Sans"


def box(ax, x, y, w, h, title, body, accent=INK, fill="#fffdf8", dashed=False):
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            facecolor=fill,
            edgecolor=LINE,
            linewidth=0.9,
            linestyle=(0, (3, 2)) if dashed else "solid",
            zorder=2,
        )
    )
    ax.add_patch(Rectangle((x, y + h - 0.012), w, 0.012, facecolor=accent, edgecolor="none", zorder=3))
    ax.text(x + 0.012, y + h - 0.035, title, fontsize=8.6, color=INK, weight="bold", va="top", zorder=4)
    ax.text(
        x + 0.012,
        y + h - 0.068,
        body,
        fontsize=6.9,
        color=MUTED,
        va="top",
        linespacing=1.45,
        zorder=4,
        wrap=True,
    )


def arrow(ax, p1, p2, color=INK, style="-|>", lw=0.9, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            p1,
            p2,
            arrowstyle=style,
            mutation_scale=8,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=2,
            shrinkB=2,
            zorder=5,
        )
    )


def main() -> None:
    metrics = json.loads((ROOT / "results" / "metrics.json").read_text())
    auroc = metrics["models"]["lgbm_fused"]["auroc"]

    fig, ax = plt.subplots(figsize=(13.2, 7.4))
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.955, "GlucoTwin", fontsize=17, color=INK, weight="bold", va="top")
    ax.text(
        0.155,
        0.949,
        "a digital twin for type 2 diabetes  /  system architecture",
        fontsize=8.4,
        color=MUTED,
        va="top",
    )
    ax.plot([0.03, 0.97], [0.918, 0.918], color=LINE, lw=0.7)

    # row 1: data generation
    ax.text(0.03, 0.895, "1  DATA", fontsize=6.6, color=CLAY, weight="bold", va="top")
    box(
        ax,
        0.03,
        0.745,
        0.205,
        0.135,
        "EHR generator",
        "1,500 synthetic Indian T2D records\nlabs, meds, TCF7L2, prior hypos\nseeded, correlated fields",
        accent=CLAY,
    )
    box(
        ax,
        0.255,
        0.745,
        0.215,
        0.135,
        "CGM + wearable simulator",
        "14 days at 5 min, 6.05M readings\nmeal model, sleep, stress, walks\nEHR-derived physiology",
        accent=CLAY,
    )
    box(
        ax,
        0.49,
        0.745,
        0.19,
        0.135,
        "Optional external check",
        "CGMacros / Shanghai T2DM\nmanual download only\nnot part of the pipeline",
        accent=MUTED,
        dashed=True,
        fill="#f2eee5",
    )

    # row 2: feature pipeline
    ax.text(0.03, 0.705, "2  FEATURES", fontsize=6.6, color=OCHRE, weight="bold", va="top")
    box(
        ax,
        0.03,
        0.535,
        0.28,
        0.15,
        "Alignment and cleaning",
        "5-minute grid, gaps interpolated\nmissingness kept as a feature\npatient-level train / val / test split",
        accent=OCHRE,
    )
    box(
        ax,
        0.33,
        0.535,
        0.32,
        0.15,
        "Feature builder  (src/features.py)",
        "55 features: 19 CGM, 13 wearable, 3 clock, 20 EHR\n2-hour label: 30 min above 180 mg/dL\nforecast targets at +30, +60, +120 min",
        accent=OCHRE,
    )
    box(
        ax,
        0.67,
        0.535,
        0.30,
        0.15,
        "Sequence tensor",
        "6 hours x 5 channels at 10 min\nlast 36 steps for the neural branch\nglucose, HR, HRV, steps, sleep stage",
        accent=OCHRE,
    )

    # row 3: models
    ax.text(0.03, 0.495, "3  MODELS", fontsize=6.6, color=SAGE, weight="bold", va="top")
    box(
        ax,
        0.03,
        0.30,
        0.30,
        0.17,
        "LightGBM, fused  (deployed)",
        f"250 trees, 55 features, isotonic calibration\ntest AUROC {auroc:.3f}, 78% sensitivity\nat 0.45 false alarms per patient-day",
        accent=SAGE,
        fill="#fbfaf4",
    )
    box(
        ax,
        0.35,
        0.30,
        0.27,
        0.17,
        "Two-branch fusion net",
        "dilated TCN + MLP, 2 shared heads\nnumpy, gradient-checked\ntest AUROC 0.950, kept as comparison",
        accent=SAGE,
    )
    box(
        ax,
        0.64,
        0.30,
        0.33,
        0.17,
        "Forecast and explanation",
        "3 LightGBM regressors for +30 / +60 / +120 min\nSHAP for the tree model, segment attribution\nfor the network, plain-language factors",
        accent=SAGE,
    )

    # row 4: twin / api / app
    ax.text(0.03, 0.26, "4  PRODUCT", fontsize=6.6, color=SLATE, weight="bold", va="top")
    box(
        ax,
        0.03,
        0.075,
        0.275,
        0.17,
        "Digital twin state  (src/twin.py)",
        "EHR + fitted parameters from CGM\nlatest 6-hour window, risk, trajectory\nalerts and replay checkpoints",
        accent=SLATE,
    )
    box(
        ax,
        0.325,
        0.075,
        0.20,
        0.17,
        "What-if engine",
        "reruns the patient's own model\nwalk, carbs, snack, sleep,\nmetformin, with effect sizes",
        accent=SLATE,
    )
    box(
        ax,
        0.545,
        0.075,
        0.185,
        0.17,
        "FastAPI service",
        "/patients, /twin, /predict,\n/whatif, /metrics, /health",
        accent=SLATE,
    )
    box(
        ax,
        0.75,
        0.075,
        0.22,
        0.17,
        "Clinician dashboard",
        "cohort triage, twin workspace,\nwhy-this-alert, scenario panel,\nalert timeline, live replay",
        accent=SLATE,
        fill="#fbfaf4",
    )
    ax.add_patch(
        Rectangle((0.75, 0.012), 0.22, 0.045, facecolor="#f2eee5", edgecolor=LINE, linewidth=0.9, zorder=2)
    )
    ax.text(
        0.86,
        0.0345,
        "React + Convex: acknowledgements and notes",
        fontsize=6.2,
        color=MUTED,
        ha="center",
        va="center",
        zorder=4,
    )

    # arrows
    arrow(ax, (0.235, 0.812), (0.255, 0.812), color=MUTED)
    arrow(ax, (0.3625, 0.745), (0.3625, 0.69), color=MUTED)
    arrow(ax, (0.17, 0.745), (0.17, 0.69), color=MUTED)
    arrow(ax, (0.26, 0.61), (0.33, 0.61), color=MUTED)
    arrow(ax, (0.65, 0.61), (0.67, 0.61), color=MUTED)
    arrow(ax, (0.18, 0.535), (0.18, 0.475), color=MUTED)
    arrow(ax, (0.48, 0.535), (0.48, 0.475), color=MUTED)
    arrow(ax, (0.80, 0.535), (0.80, 0.475), color=MUTED)
    arrow(ax, (0.17, 0.30), (0.17, 0.25), color=MUTED)
    arrow(ax, (0.42, 0.30), (0.42, 0.25), color=MUTED)
    arrow(ax, (0.80, 0.30), (0.80, 0.25), color=MUTED)
    arrow(ax, (0.305, 0.16), (0.325, 0.16), color=MUTED)
    arrow(ax, (0.525, 0.16), (0.545, 0.16), color=MUTED)
    arrow(ax, (0.73, 0.16), (0.75, 0.16), color=MUTED)
    arrow(ax, (0.86, 0.30), (0.86, 0.25), color=MUTED)

    # feedback loop from twin to what-if and from dashboard back to twin
    arrow(ax, (0.305, 0.115), (0.325, 0.115), color=SLATE, rad=-0.35)
    ax.annotate(
        "scenario curves return to the twin and to the clinician,\nnever to an automatic treatment change",
        xy=(0.72, 0.055),
        xytext=(0.36, 0.045),
        fontsize=6.4,
        color=MUTED,
        ha="left",
        va="center",
    )

    ax.text(
        0.03,
        0.015,
        "All modelling runs on one laptop CPU.  Synthetic data only.  DPDP and HIPAA by design.  A clinician makes every decision.",
        fontsize=6.3,
        color=MUTED,
    )

    DOCS.mkdir(exist_ok=True)
    fig.savefig(DOCS / "architecture.png", dpi=300, facecolor=PAPER, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(DOCS / "architecture.pdf", facecolor=PAPER, bbox_inches="tight", pad_inches=0.15)
    print(f"[architecture] wrote {DOCS / 'architecture.png'} and .pdf")


if __name__ == "__main__":
    main()
