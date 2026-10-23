"""Render the dashboard plates used in the slides and the video.

This sandbox has no browser binary, so these images are drawn directly from the
same exported JSON the React dashboard reads (public/data/). They use the same
palette, the same numbers and the same layout logic as the live screens. They
are not browser screenshots, and the README says so.

Outputs
-------
docs/screenshots/cohort.png
docs/screenshots/twin.png
docs/screenshots/why-alert.png
docs/screenshots/whatif.png
docs/screenshots/alerts.png

Run:  python src/make_screenshots.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle, Wedge

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "public" / "data"
OUT = ROOT / "docs" / "screenshots"

PAPER = "#f6f2e9"
CARD = "#fffdf8"
INK = "#1c1b19"
MUTED = "#7a7468"
LINE = "#ded7c9"
CLAY = "#b4552d"
OCHRE = "#b08a4f"
SAGE = "#6f8060"
SLATE = "#4f6b7a"

STATUS = {
    "critical": ("Critical", CLAY),
    "watch": ("Watch", OCHRE),
    "stable": ("Stable", SAGE),
}
PATIENT_ID = "GT-0568"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": LINE,
        "figure.facecolor": PAPER,
        "savefig.facecolor": PAPER,
    }
)


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def base_figure(width=16.0, height=10.0):
    fig = plt.figure(figsize=(width, height))
    fig.patch.set_facecolor(PAPER)
    return fig


def header(fig, title: str, subtitle: str, right: str = "") -> None:
    fig.text(0.045, 0.955, "GlucoTwin", fontsize=13, weight="bold", va="top")
    fig.text(0.045, 0.921, title, fontsize=10.5, color=MUTED, va="top")
    fig.text(0.045, 0.885, subtitle, fontsize=17, va="top")
    if right:
        fig.text(0.955, 0.885, right, fontsize=9, color=MUTED, ha="right", va="top")
    fig.add_artist(plt.Line2D([0.045, 0.955], [0.862, 0.862], color=LINE, lw=0.8, figure=fig))


def footer(fig, text: str) -> None:
    fig.text(0.045, 0.030, text, fontsize=7.4, color=MUTED)
    fig.add_artist(plt.Line2D([0.045, 0.955], [0.055, 0.055], color=LINE, lw=0.7, figure=fig))


def frame(ax, color=LINE, fill=CARD):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor=fill, edgecolor=color, lw=0.8, zorder=-2, clip_on=False))


def clock(minutes: float) -> str:
    m = int(round(minutes)) % 1440
    return f"{m // 60:02d}:{m % 60:02d}"


def clean_axes(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(LINE)
    ax.spines["bottom"].set_color(LINE)
    ax.grid(True, axis=grid_axis, color=LINE, lw=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8, length=0)


# ---------------------------------------------------------------- 1 cohort


def cohort_plate(cohort: dict) -> None:
    rows = cohort["patients"]
    counts = {"critical": 0, "watch": 0, "stable": 0}
    for row in rows:
        counts[row["status"]] += 1

    fig = base_figure()
    header(
        fig,
        "CLINIC / COHORT",
        "Panel of 60, sorted by two-hour risk",
        f"reviewed at 20:00  ·  {cohort['cohort']['n_patients']:,} patients in the synthetic cohort",
    )

    stats = [
        ("Critical", counts["critical"], CLAY, "predicted event in the next 2 h"),
        ("Watch", counts["watch"], OCHRE, "borderline, recheck this shift"),
        ("Stable", counts["stable"], SAGE, "no event predicted"),
    ]
    for i, (label, value, color, note) in enumerate(stats):
        x = 0.045 + i * 0.14
        fig.text(x, 0.815, label.upper(), fontsize=7.5, color=color, weight="bold")
        fig.text(x, 0.775, str(value), fontsize=26, va="top")
        fig.text(x, 0.727, note, fontsize=7.4, color=MUTED)

    fig.text(0.62, 0.815, "MEAN HBA1C", fontsize=7, color=MUTED)
    fig.text(0.62, 0.785, f"{cohort['cohort']['mean_hba1c']:.1f}%", fontsize=12)
    fig.text(0.72, 0.815, "TIR, PANEL MEDIAN", fontsize=7, color=MUTED)
    median_tir = float(np.median([r["tir_in_range"] for r in rows]))
    fig.text(0.72, 0.785, f"{median_tir * 100:.0f}%", fontsize=12)
    fig.text(0.84, 0.815, "ALERTS, LAST 72 H", fontsize=7, color=MUTED)
    fig.text(0.84, 0.785, f"{sum(r['alerts_72h'] for r in rows)}", fontsize=12)

    ax = fig.add_axes([0.045, 0.09, 0.91, 0.63])
    frame(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    cols = [
        (0.016, "PATIENT"),
        (0.115, "AGE / SEX"),
        (0.205, "HBA1C"),
        (0.275, "GLUCOSE"),
        (0.355, "TIME IN RANGE"),
        (0.495, "2-HOUR RISK"),
        (0.77, "ALERTS 72 H"),
        (0.865, "STATUS"),
    ]
    for x, label in cols:
        ax.text(x, 0.945, label, fontsize=7, color=MUTED, va="center")

    shown = rows[:16]
    top = 0.895
    step = 0.053
    for i, row in enumerate(shown):
        y = top - i * step
        if i % 2 == 1:
            ax.add_patch(Rectangle((0.008, y - 0.026), 0.984, 0.05, facecolor="#f3eee4", edgecolor="none", zorder=-1))
        ax.text(0.016, y, row["id"], fontsize=8.4, va="center")
        ax.text(0.115, y, f"{row['age']} / {row['sex']}", fontsize=8.2, color=MUTED, va="center")
        ax.text(0.205, y, f"{row['hba1c']:.1f}", fontsize=8.2, va="center")
        ax.text(0.275, y, f"{row['glucose']:.0f}", fontsize=8.2, va="center")
        ax.add_patch(Rectangle((0.355, y - 0.007), 0.11, 0.014, facecolor="#e9e3d6", edgecolor="none"))
        ax.add_patch(Rectangle((0.355, y - 0.007), 0.11 * row["tir_in_range"], 0.014, facecolor=SAGE, edgecolor="none", alpha=0.75))
        ax.text(0.472, y, f"{row['tir_in_range'] * 100:.0f}%", fontsize=7.6, color=MUTED, va="center")
        ax.add_patch(Rectangle((0.495, y - 0.006), 0.19, 0.012, facecolor="#e9e3d6", edgecolor="none"))
        ax.add_patch(Rectangle((0.495, y - 0.006), 0.19 * row["risk"], 0.012, facecolor=CLAY if row["status"] == "critical" else (OCHRE if row["status"] == "watch" else SAGE), edgecolor="none"))
        ax.text(0.695, y, f"{row['risk'] * 100:.0f}%", fontsize=7.8, va="center")
        ax.text(0.77, y, str(row["alerts_72h"]), fontsize=8, color=MUTED, va="center")
        label, color = STATUS[row["status"]]
        ax.add_patch(FancyBboxPatch((0.865, y - 0.017), 0.10, 0.034, boxstyle="round,pad=0.002,rounding_size=0.012", facecolor="white", edgecolor=color, lw=0.8))
        ax.add_patch(plt.Circle((0.877, y), 0.0035, color=color, transform=ax.transData))
        ax.text(0.888, y, label, fontsize=7.4, color=color, va="center")
    ax.text(0.016, 0.015, f"showing 16 of {len(rows)}  ·  every row is synthetic  ·  a clinician opens the twin before acting", fontsize=7.2, color=MUTED)

    footer(fig, "2-hour risk from the deployed LightGBM model. Research prototype, not a medical device.")
    fig.savefig(OUT / "cohort.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ 2 twin


def gauge(ax, risk: float) -> None:
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-0.18, 1.12)
    ax.axis("off")
    ax.add_patch(Wedge((0, 0), 1.0, 0, 180, width=0.18, facecolor="#e9e3d6", edgecolor="none"))
    if risk > 0:
        ax.add_patch(Wedge((0, 0), 1.0, 180 - 180 * risk, 180, width=0.18, facecolor=CLAY, edgecolor="none"))
    ax.text(0, 0.30, f"{risk * 100:.0f}%", fontsize=27, ha="center", va="center", color=CLAY if risk > 0.7 else INK)
    ax.text(0, 0.10, "event in next 2 h", fontsize=8, ha="center", color=MUTED)
    for tick in (0.0, 0.5, 1.0):
        angle = np.deg2rad(180 - 180 * tick)
        ax.plot([0.84 * np.cos(angle), 0.95 * np.cos(angle)], [0.84 * np.sin(angle), 0.95 * np.sin(angle)], color=LINE, lw=0.8)
        ax.text(1.02 * np.cos(angle), 1.02 * np.sin(angle), f"{tick * 100:.0f}", fontsize=6.5, color=MUTED, ha="center", va="center")


def twin_plate(p: dict) -> None:
    summary = p["summary"]
    ehr = p["ehr"]
    fig = base_figure()
    header(
        fig,
        f"PANEL / {p['id']}  ·  {ehr['age']} y  ·  {ehr['sex']}  ·  T2D {ehr['years_since_dx']:.0f} years  ·  {'insulin' if ehr['insulin'] else 'oral therapy'}",
        "Twin workspace",
        "last CGM reading 20:00  ·  synthetic patient",
    )

    label, color = STATUS[summary["status"]]
    fig.text(0.955, 0.921, label.upper(), fontsize=9, color=color, weight="bold", ha="right")

    # --- CGM trace + forecast
    ax = fig.add_axes([0.055, 0.40, 0.56, 0.40])
    frame(ax)
    clean_axes(ax)
    recent = np.array(p["series"]["recent"]["values"], dtype=float)
    now = summary["now_minute"]
    view = recent[-96:]
    start = now - len(view) * 5
    xs = np.arange(start, now, 5)[-len(view):]
    ax.plot(xs, view, color=INK, lw=1.25, solid_joinstyle="round")
    ax.axhline(180, color=CLAY, lw=0.9, ls=(0, (4, 3)))
    ax.text(start + 4, 183, "180 mg/dL", fontsize=7, color=CLAY, va="bottom")
    ax.axvspan(now, now + 120, color=CLAY, alpha=0.05, lw=0)

    cp = p["replay_checkpoints"][-1]["forecast"]
    fx = np.array([now + i * 5 for i in range(len(cp["mean"]))])
    ax.fill_between(fx, cp["lo"], cp["hi"], color=CLAY, alpha=0.16, lw=0)
    ax.plot(fx, cp["mean"], color=CLAY, lw=1.3, ls="-")
    ax.plot([now], [summary["glucose"]], marker="o", ms=4.5, color=INK, zorder=5)
    ax.text(now + 4, summary["glucose"] + 12, f"{summary['glucose']:.0f}", fontsize=8, color=INK)

    ax.set_xlim(start, now + 118)
    ax.set_ylim(60, max(300, float(np.nanmax(cp["hi"])) + 20))
    ticks = [start + i * 60 for i in range(9) if start + i * 60 <= now + 120]
    ax.set_xticks(ticks)
    ax.set_xticklabels([clock(t) for t in ticks], fontsize=7.5)
    ax.set_ylabel("mg/dL", fontsize=7.5)
    ax.text(0.005, 1.03, "CGM, last 8 hours", transform=ax.transAxes, fontsize=7.2, color=MUTED)
    ax.text(0.99, 1.03, "shaded: forecast + 95% band", transform=ax.transAxes, fontsize=7.2, color=CLAY, ha="right")

    # --- gauge and ranges
    axg = fig.add_axes([0.665, 0.60, 0.29, 0.20])
    gauge(axg, float(summary["risk"]))

    axr = fig.add_axes([0.685, 0.48, 0.25, 0.055])
    axr.set_xlim(0, 1)
    axr.set_ylim(0, 1)
    axr.axis("off")
    tir = p["summary"]["tir"]
    segments = [
        ("in range", tir["in_range"], SAGE),
        ("above 180", tir["above"], CLAY),
        ("below 70", tir["below"], SLATE),
    ]
    x = 0.0
    for _name, share, col in segments:
        if share > 0:
            axr.add_patch(Rectangle((x, 0.35), share, 0.30, facecolor=col, edgecolor="none", alpha=0.85))
            x += share
    axr.add_patch(Rectangle((0, 0.35), 1, 0.30, facecolor="none", edgecolor=LINE, lw=0.7))
    axr.text(0, 1.25, "TIME IN RANGE, 24 H", fontsize=7, color=MUTED, va="bottom")
    axr.text(1.0, 1.25, f"{tir['in_range'] * 100:.1f}% in range", fontsize=7.4, color=INK, ha="right", va="bottom")
    axr.text(0, -0.35, f"mean {tir['mean']:.0f} mg/dL  ·  above {tir['above'] * 100:.0f}%  ·  below {tir['below'] * 100:.0f}%", fontsize=7, color=MUTED, va="top")

    # --- key numbers
    fig.text(0.685, 0.40, "NOW", fontsize=7, color=MUTED)
    fig.text(0.685, 0.355, f"{summary['glucose']:.0f}", fontsize=22)
    fig.text(0.735, 0.372, "mg/dL", fontsize=7.5, color=MUTED)
    fig.text(0.685, 0.325, f"{'falling' if summary['trend_30m'] < -4 else ('rising' if summary['trend_30m'] > 4 else 'steady')} {abs(summary['trend_30m']):.0f} in 30 min",
             fontsize=7.6, color=SAGE if summary["trend_30m"] < 0 else CLAY)

    fig.text(0.80, 0.40, "PEAK RISK, NEXT 24 H", fontsize=7, color=MUTED)
    fig.text(0.80, 0.355, f"{summary['risk_max_24h'] * 100:.0f}%", fontsize=22)
    fig.text(0.855, 0.372, "of windows", fontsize=7.2, color=MUTED)

    # --- wearable panels
    day = p["series"]["replay_day"]
    hours = np.arange(len(day["glucose"])) * 5 / 60

    ax1 = fig.add_axes([0.055, 0.10, 0.275, 0.15])
    frame(ax1)
    clean_axes(ax1)
    ax1.plot(hours, day["hrv"], color=SLATE, lw=1.1)
    ax1.set_xlim(0, 24)
    ax1.set_xticks([0, 6, 12, 18, 24])
    ax1.set_ylabel("ms", fontsize=7)
    ax1.text(0.0, 1.08, "HRV (RMSSD), today", transform=ax1.transAxes, fontsize=7.2, color=MUTED)

    ax2 = fig.add_axes([0.365, 0.10, 0.275, 0.15])
    frame(ax2)
    clean_axes(ax2)
    stage = np.array(day["stage"])
    ax2.fill_between(hours, 0, np.where(stage > 0, 1, 0), step="mid", color="#dfe4d7", lw=0)
    ax2.fill_between(hours, 0, np.where(stage == 2, 1, 0), step="mid", color=SAGE, lw=0, alpha=0.55)
    ax2.set_yticks([])
    ax2.set_xlim(0, 24)
    ax2.set_xticks([0, 6, 12, 18, 24])
    ax2.text(0.0, 1.08, "sleep stages, today", transform=ax2.transAxes, fontsize=7.2, color=MUTED)
    ax2.text(0.99, 0.55, "deep", fontsize=6.8, color=SAGE, ha="right")

    ax3 = fig.add_axes([0.675, 0.10, 0.28, 0.15])
    frame(ax3)
    clean_axes(ax3)
    steps = np.array(day["steps"], dtype=float)
    steps = np.pad(steps, (0, max(0, 288 - len(steps))))
    block = steps.reshape(72, 4).sum(axis=1)
    ax3.bar(np.arange(72) * 20 / 60, block, width=0.22, color=OCHRE, alpha=0.8, linewidth=0)
    ax3.set_xlim(0, 24)
    ax3.set_xticks([0, 6, 12, 18, 24])
    ax3.text(0.0, 1.08, "steps per 20 min, today", transform=ax3.transAxes, fontsize=7.2, color=MUTED)

    footer(fig, "Forecast band is the LightGBM quantile envelope over the next 2 hours. Simulated patient, DPDP and HIPAA by design.")
    fig.savefig(OUT / "twin.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------- 3 why alert


def why_plate(p: dict) -> None:
    factors = p["replay_checkpoints"][-1]["factors"]
    summary = p["summary"]
    fig = base_figure()
    header(
        fig,
        f"PANEL / {p['id']}  ·  WHY THIS ALERT",
        "Five factors behind the 20:00 alert",
        f"risk {summary['risk'] * 100:.0f}%  ·  glucose {summary['glucose']:.0f} mg/dL",
    )

    ax = fig.add_axes([0.30, 0.30, 0.62, 0.48])
    frame(ax)
    clean_axes(ax, grid_axis="x")
    labels = [f["label"] for f in factors][::-1]
    values = [f["contribution"] for f in factors][::-1]
    colors = [CLAY if v > 0 else SAGE for v in values]
    ypos = np.arange(len(labels))
    ax.barh(ypos, values, color=colors, height=0.42, alpha=0.9)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color=INK, lw=0.7)
    for y, v in zip(ypos, values):
        ax.text(v + (0.06 if v >= 0 else -0.06), y, f"{v:+.2f}", fontsize=8, color=MUTED, va="center", ha="left" if v >= 0 else "right")
    ax.set_xlabel("SHAP contribution to the log-odds of an event", fontsize=8)
    ax.set_xlim(min(values + [0]) * 1.35, max(values + [0]) * 1.35)

    fig.text(0.045, 0.72, "IN PLAIN LANGUAGE", fontsize=7.5, color=MUTED)
    lines = [
        "Glucose has been high for most of the",
        "afternoon and is still above target.",
        "The evening meal is the next big input.",
        "",
        "No sensor gap and no low reading, so this",
        "is not a compression-low artefact.",
    ]
    for i, line in enumerate(lines):
        fig.text(0.045, 0.68 - i * 0.038, line, fontsize=9.5, color=INK if line else MUTED)

    fig.text(0.045, 0.40, "WHAT TO CHECK FIRST", fontsize=7.5, color=MUTED)
    checks = ["missed or late dose", "late, larger dinner", "sensor lag after a meal"]
    for i, item in enumerate(checks):
        fig.text(0.045, 0.36 - i * 0.034, f"{i + 1}.  {item}", fontsize=9, color=MUTED)

    fig.text(0.045, 0.22, "This panel explains the model. It does not decide treatment.", fontsize=8, color=CLAY)

    footer(fig, "SHAP values from the deployed tree model, merged into plain-language groups. A clinician decides what to do.")
    fig.savefig(OUT / "why-alert.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 4 whatif


def whatif_plate(p: dict) -> None:
    whatif = p["whatif"]
    minutes = np.array(whatif["minutes"]) + whatif["start_min"]
    baseline = whatif["baseline"]
    fig = base_figure()
    header(
        fig,
        f"PANEL / {p['id']}  ·  WHAT-IF",
        "Same patient, same evening, five scenarios",
        "simulation for decision support, not medical advice",
    )

    ax = fig.add_axes([0.055, 0.30, 0.60, 0.50])
    frame(ax)
    clean_axes(ax)
    ax.plot(minutes, baseline["curve"], color=INK, lw=1.7, label="current evening")
    picks = [
        ("walk_after_dinner", SAGE, "-"),
        ("swap_rice", SLATE, "-"),
        ("metformin_up", OCHRE, (0, (5, 2))),
    ]
    for key, color, style in picks:
        scenario = whatif["scenarios"][key]
        ax.plot(minutes, scenario["curve"], color=color, lw=1.2, ls=style, label=scenario["label"])
    ax.axhline(180, color=CLAY, lw=0.9, ls=(0, (4, 3)))
    ax.text(minutes[2], 183, "180 mg/dL", fontsize=7, color=CLAY)
    ax.set_xlim(minutes[0], minutes[-1])
    ticks = [minutes[i] for i in range(0, len(minutes), 24)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([clock(t) for t in ticks], fontsize=8)
    ax.set_ylabel("mg/dL", fontsize=8)
    ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=1)
    ax.set_title("glucose after the current plan and after each scenario", fontsize=8.5, color=MUTED, loc="left", pad=10)

    # scenario table
    ax2 = fig.add_axes([0.045, 0.075, 0.91, 0.17])
    frame(ax2)
    ax2.axis("off")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    cols = [(0.012, "SCENARIO"), (0.30, "PEAK"), (0.42, "MIN > 180"), (0.55, "2-H RISK"), (0.68, "CHANGE")]
    for x, name in cols:
        ax2.text(x, 0.86, name, fontsize=7, color=MUTED)

    rows = [("current evening", baseline, None)] + [
        (whatif["scenarios"][key]["label"], whatif["scenarios"][key], key)
        for key in ("walk_after_dinner", "swap_rice", "skip_snack", "sleep_7h", "metformin_up")
    ]
    for i, (label, row, _key) in enumerate(rows):
        y = 0.64 - i * 0.115
        ax2.text(0.012, y, label, fontsize=8, color=INK if i == 0 else MUTED)
        ax2.text(0.30, y, f"{row['peak']:.0f}", fontsize=8, color=INK if i == 0 else MUTED)
        ax2.text(0.42, y, f"{row['minutes_above_180']:.0f} min", fontsize=8, color=INK if i == 0 else MUTED)
        ax2.text(0.55, y, f"{row['prob'] * 100:.0f}%", fontsize=8, color=INK if i == 0 else MUTED)
        if i > 0:
            delta = row["delta_prob"]
            tone = SAGE if delta < -0.02 else MUTED
            ax2.text(0.68, y, f"{delta * 100:+.0f} pts", fontsize=8, color=tone)
        else:
            ax2.text(0.68, y, "reference", fontsize=8, color=MUTED)

    footer(fig, "Scenarios rerun this patient's fitted physiology. They are not prescriptions and not predictions of what will happen.")
    fig.savefig(OUT / "whatif.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------- 5 alerts


def alerts_plate(p: dict) -> None:
    trace = np.array(p["risk_trace"]["values"], dtype=float)
    step_min = p["risk_trace"]["step_min"]
    now_step = p["summary"]["now_minute"] // step_min
    window = trace[max(0, now_step - 3 * 96) : now_step + 1]
    xs = np.arange(len(window)) * step_min
    offset = p["summary"]["now_minute"] - len(window) * step_min

    fig = base_figure()
    header(
        fig,
        f"PANEL / {p['id']}  ·  ALERTS",
        "Seventy-two hours of risk and ten warnings",
        "10 alerts in the last 3 days  ·  9 matched a real event",
    )

    ax = fig.add_axes([0.06, 0.44, 0.88, 0.34])
    frame(ax)
    clean_axes(ax)
    ax.plot(xs, window, color=INK, lw=1.2)
    ax.fill_between(xs, 0, window, color=CLAY, alpha=0.08)
    ax.axhline(0.995, color=CLAY, lw=0.9, ls=(0, (4, 3)))
    ax.text(4, 1.002, "alert threshold", fontsize=7, color=CLAY)
    ax.set_ylim(0, 1.08)
    ax.set_xlim(0, xs[-1])
    ax.set_ylabel("2-hour risk", fontsize=8)

    alerts = p["alerts"]
    for alert in alerts:
        minute = alert["day"] * 1440 + alert["minute_of_day"] - offset
        if minute < 0 or minute > xs[-1]:
            continue
        color = CLAY if alert["matched"] else OCHRE
        ax.plot([minute], [min(alert["risk"], 1.06)], marker="o", ms=4.6, color=color, zorder=5)
        if alert["matched"] and alert["lead_min"]:
            ax.annotate(
                f"{alert['lead_min']} min",
                xy=(minute, min(alert["risk"], 1.06)),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                fontsize=6.4,
                color=CLAY,
            )
    ticks = [i for i in range(0, xs[-1] + 1, 360)]
    ax.set_xticks(ticks)
    labels = ["now" if t == ticks[-1] else f"-{((xs[-1] - t) / 60):.0f} h" for t in ticks]
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.text(0.005, 1.05, "risk trace, 15-minute checkpoints", transform=ax.transAxes, fontsize=7.2, color=MUTED)

    # glucose strip underneath
    axg = fig.add_axes([0.06, 0.15, 0.88, 0.22])
    frame(axg)
    clean_axes(axg)
    day = p["series"]["replay_day"]
    g = np.array(day["glucose"], dtype=float)
    axg.plot(np.arange(len(g)) * 5, g, color=SLATE, lw=1.0)
    axg.axhline(180, color=CLAY, lw=0.8, ls=(0, (4, 3)))
    axg.set_ylim(60, max(260, float(np.nanmax(g)) + 15))
    axg.set_xlim(0, 1440)
    axg.set_xticks([0, 360, 720, 1080, 1440])
    axg.set_xticklabels(["00:00", "06:00", "12:00", "18:00", "24:00"], fontsize=7.5)
    axg.set_ylabel("mg/dL", fontsize=7.5)
    axg.text(0.005, 1.04, f"glucose, replay day {day['day'] + 1}", transform=axg.transAxes, fontsize=7.2, color=MUTED)

    fig.text(0.06, 0.095, "filled dot: matched a real hyperglycemic run   ·   hollow dot: unmatched warning   ·   number above: warning lead time", fontsize=7.4, color=MUTED)

    footer(fig, "Alerts are first crossings of the operating threshold with a 2-hour cooldown. Simulated patient.")
    fig.savefig(OUT / "alerts.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cohort = load(DATA / "cohort.json")
    patient = load(DATA / "patients" / f"{PATIENT_ID}.json")

    cohort_plate(cohort)
    twin_plate(patient)
    why_plate(patient)
    whatif_plate(patient)
    alerts_plate(patient)
    print(f"[screenshots] wrote 5 plates to {OUT}")


if __name__ == "__main__":
    main()
