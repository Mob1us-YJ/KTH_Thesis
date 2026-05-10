"""
Generate publication-quality XRF55 Stage-B results figure (Fig. 5.5).
Data source: Tables 5.5 and 5.6 in the thesis (best-over-3-trials values).

Panel (a): Grouped bar chart — validation accuracy and macro-F1 for
           three architecture families (plain, ResNet, multiscale) at w=96.
Panel (b): Per-class recall for the best multiscale trial on the
           6-class XRF55 validation set.

Output: figures/xrf55_stageb_results.pdf
Run from thesis root: python figures/gen_xrf55_stageb_figure.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

# ── Style ─────────────────────────────────────────────────────────────────────
KTH_BLUE  = "#1954a6"
KTH_LBLUE = "#24a0d8"
DARK_GRAY = "#3a3a3a"
MED_GRAY  = "#6e6e6e"
ORANGE    = "#c44601"
OUT_DIR   = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          9,
    "axes.labelsize":     9,
    "axes.titlesize":     9.5,
    "legend.fontsize":    8,
    "xtick.labelsize":    7.5,
    "ytick.labelsize":    7.5,
    "text.usetex":        False,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})

# ── Data (from Table 5.5 and Table 5.6) ───────────────────────────────────────
families   = ["Plain", "ResNet", "Multiscale"]
val_acc    = [0.8990, 0.9276, 0.9343]
macro_f1   = [0.8673, 0.9093, 0.9268]

classes       = ["Sitting", "Reaching", "Turning", "Bending", "Waving", "Using\nphone"]
class_recall  = [1.000, 0.909, 0.919, 0.924, 0.955, 0.909]


def draw_stageb():
    fig = plt.figure(figsize=(7.2, 3.0))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.48,
                            left=0.07, right=0.97, bottom=0.15, top=0.84)

    # ── (a) Architecture family comparison ───────────────────────────────────
    ax1 = fig.add_subplot(gs[0])

    x     = np.arange(len(families))
    width = 0.34

    bars_acc = ax1.bar(x - width / 2, val_acc,  width, label="Val Acc",
                       color=KTH_BLUE,  alpha=0.88, zorder=3)
    bars_f1  = ax1.bar(x + width / 2, macro_f1, width, label="Macro F1",
                       color=KTH_LBLUE, alpha=0.88, zorder=3)

    # Value labels
    for bar in bars_acc:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                 f"{bar.get_height():.3f}", ha="center", va="bottom",
                 fontsize=6.5, color=DARK_GRAY)
    for bar in bars_f1:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                 f"{bar.get_height():.3f}", ha="center", va="bottom",
                 fontsize=6.5, color=DARK_GRAY)

    # Best-family annotation
    best_idx = int(np.argmax(val_acc))
    ax1.annotate("best", xy=(x[best_idx] - width / 2, val_acc[best_idx]),
                 xytext=(x[best_idx] - width / 2 - 0.28, val_acc[best_idx] + 0.018),
                 fontsize=6, color=ORANGE,
                 arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=0.7))

    ax1.set_xticks(x)
    ax1.set_xticklabels(families)
    ax1.set_ylabel("Score")
    ax1.set_ylim(0.80, 0.975)
    ax1.yaxis.set_major_locator(plt.MultipleLocator(0.02))
    ax1.grid(axis="y", lw=0.4, alpha=0.5, zorder=0)
    ax1.legend(loc="lower right", framealpha=0.85)
    ax1.set_title("(a)  Architecture family — XRF55 Stage B",
                  loc="left", fontsize=8.5, pad=6, color=DARK_GRAY)

    # ── (b) Per-class recall (best multiscale trial) ──────────────────────────
    ax2 = fig.add_subplot(gs[1])

    colors = [KTH_BLUE if r < 0.95 else ORANGE for r in class_recall]
    bars_r = ax2.bar(np.arange(len(classes)), class_recall, color=colors,
                     alpha=0.88, zorder=3)

    for bar, r in zip(bars_r, class_recall):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                 f"{r:.3f}", ha="center", va="bottom",
                 fontsize=6.5, color=DARK_GRAY)

    # Reference line at mean recall
    mean_r = np.mean(class_recall)
    ax2.axhline(mean_r, color=MED_GRAY, lw=0.8, ls="--", zorder=2)
    ax2.text(len(classes) - 0.5, mean_r + 0.003,
             f"mean={mean_r:.3f}", ha="right", va="bottom",
             fontsize=6, color=MED_GRAY)

    ax2.set_xticks(np.arange(len(classes)))
    ax2.set_xticklabels(classes, fontsize=7)
    ax2.set_ylabel("Recall")
    ax2.set_ylim(0.84, 1.055)
    ax2.yaxis.set_major_locator(plt.MultipleLocator(0.02))
    ax2.grid(axis="y", lw=0.4, alpha=0.5, zorder=0)
    ax2.set_title("(b)  Per-class recall — best multiscale model",
                  loc="left", fontsize=8.5, pad=6, color=DARK_GRAY)

    out = os.path.join(OUT_DIR, "xrf55_stageb_results.pdf")
    fig.savefig(out, format="pdf")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    draw_stageb()
    print("Done.")
