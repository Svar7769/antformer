# Plot training curves from one or more runs.
#
#   python -m model.plot outputs/tf_wd1                      -> outputs/tf_wd1/curves.png
#   python -m model.plot outputs/tf_wd1 outputs/tf_wd0 --out compare.png
#
# Colour = what is measured (probe, case, loss term). Line style = which run.
# Panels: losses (with the best possible mode loss), direction accuracy per case,
#         synth accuracy per probe, P(wander) calibration, weight norm, pick/drop.

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from dataset.config import PROBES, CASES

STYLES = ["-", "--", ":", "-."]
COLORS = plt.get_cmap("tab10").colors


def load(run_dir):
    rows = [json.loads(l) for l in open(os.path.join(run_dir, "metrics.jsonl")) if l.strip()]
    keys = set().union(*rows)
    return {k: [r.get(k) for r in rows] for k in keys}


def draw(ax, m, key, color, ls):
    if key in m and any(v is not None for v in m[key]):
        pts = [(s, v) for s, v in zip(m["step"], m[key]) if v is not None]
        ax.plot(*zip(*pts), color=color, ls=ls, lw=1.4)


def legend(ax, names, title=None, loc="best"):
    handles = [Line2D([], [], color=COLORS[i % 10], lw=2) for i in range(len(names))]
    ax.legend(handles, names, fontsize=7, title=title, title_fontsize=7, loc=loc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--logx", action="store_true")
    args = ap.parse_args()

    runs = {os.path.basename(os.path.normpath(r)): load(r) for r in args.runs}
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    ax_loss, ax_dir, ax_probe, ax_cal, ax_wn, ax_ev = axes.flat

    loss_keys = ["train/loss", "val/loss", "val/loss_mode", "val/loss_dir"]
    ev_keys = ["val/pick_recall", "val/drop_recall", "val/pick_precision", "val/drop_precision"]
    for r, (name, m) in enumerate(runs.items()):
        ls = STYLES[r % len(STYLES)]
        for i, k in enumerate(loss_keys):
            draw(ax_loss, m, k, COLORS[i], ls)
        for i, c in enumerate(("target", "smell")):
            draw(ax_dir, m, f"val/dir_acc/{c}", COLORS[i], ls)
        for i, p in enumerate(PROBES):
            draw(ax_probe, m, f"synth/probe/{p}", COLORS[i % 10], ls)
        for i, c in enumerate(CASES):
            draw(ax_cal, m, f"val/p_wander/{c}", COLORS[i], ls)
        draw(ax_wn, m, "weight_norm", COLORS[0], ls)
        for i, k in enumerate(ev_keys):
            draw(ax_ev, m, k, COLORS[i], ls)

    first = next(iter(runs.values()))
    if first.get("val/ref/loss_mode_min"):
        ax_loss.axhline(first["val/ref/loss_mode_min"][0], color="gray", ls=":", lw=1)
        ax_loss.text(0.01, first["val/ref/loss_mode_min"][0], " mode loss floor", color="gray",
                     fontsize=7, va="bottom", transform=ax_loss.get_yaxis_transform())
    for y in (0.0, 0.25, 1.0):
        ax_cal.axhline(y, color="gray", ls=":", lw=0.8)

    legend(ax_loss, ["train total", "val total", "val mode", "val dir"])
    legend(ax_dir, ["target", "smell"])
    legend(ax_probe, PROBES, loc="lower right")
    legend(ax_cal, list(CASES))
    legend(ax_ev, ["pick recall", "drop recall", "pick precision", "drop precision"], loc="lower right")
    if len(runs) > 1:
        ax_wn.legend([Line2D([], [], color="k", ls=STYLES[i % len(STYLES)]) for i in range(len(runs))],
                     list(runs), fontsize=8, title="runs (line style)", title_fontsize=8)

    titles = ["loss", "val direction accuracy", "synth accuracy per probe",
              "mean P(wander) per case (true: 0, 0.25, 1)", "weight norm", "pick / drop (val)"]
    for ax, t in zip(axes.flat, titles):
        ax.set_title(t)
        ax.set_xlabel("step")
        if args.logx:
            ax.set_xscale("log")
    ax_loss.set_yscale("log")
    fig.tight_layout()

    out = args.out or os.path.join(args.runs[0], "curves.png")
    fig.savefig(out, dpi=110)
    print("saved", out)


if __name__ == "__main__":
    main()
