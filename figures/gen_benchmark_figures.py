"""
Generate publication-quality benchmark figures for Sections 5.1.
Three datasets: UT-HAR (supervised), NTU-Fi HAR (supervised),
                NTU-Human ID (SSL fine-tuning).
Five models: MLP, CNN, RNN, CNN+GRU, ViT.

Data sources:
  UT-HAR        : fig_uthar_train_test_bar.png (visual read)
  NTU-Fi HAR    : ntufi_results.json  (LeNet->CNN, BiLSTM->ViT)
  NTU-Human ID  : quick_results/*.npz  (SSL avg of two fine-tuning runs)

Output PDFs: figures/bench_uthar.pdf
             figures/bench_ntufi.pdf
             figures/bench_ntuhumanid.pdf
Run from thesis root: python figures/gen_benchmark_figures.py
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
    "xtick.labelsize":    8,
    "ytick.labelsize":    7.5,
    "text.usetex":        False,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})

MODELS = ["MLP", "CNN", "RNN", "CNN+GRU", "ViT"]

# ── Benchmark data ─────────────────────────────────────────────────────────────
# UT-HAR: read from fig_uthar_train_test_bar.png
UTHAR = {
    "train": [84.7, 99.4, 82.3, 94.5, 86.5],
    "test":  [84.0, 96.8, 82.6, 91.0, 85.0],
}

# NTU-Fi HAR: ntufi_results.json  (LeNet->CNN, BiLSTM->ViT)
NTUFI = {
    "train": [100.0, 100.0,  95.0, 100.0,  99.8],
    "test":  [ 98.1,  97.3,  85.2,  99.6, 100.0],
}

# NTU-Human ID (SSL fine-tuning): average of sup_train/test_acc1+acc2 from npz
HUMANID = {
    "train": [98.7, 100.0, 70.0, 59.4, 64.1],
    "test":  [88.3,  96.6, 59.2, 56.3, 56.8],
}


def draw_benchmark(data, title_tag, subtitle, out_name, y_lo=50.0):
    trains = data["train"]
    tests  = data["test"]
    gaps   = [round(tr - te, 1) for tr, te in zip(trains, tests)]

    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    x     = np.arange(len(MODELS))
    width = 0.34

    bars_tr = ax.bar(x - width / 2, trains, width, label="Train acc.",
                     color=KTH_BLUE,  alpha=0.88, zorder=3)
    bars_te = ax.bar(x + width / 2, tests,  width, label="Test acc.",
                     color=KTH_LBLUE, alpha=0.88, zorder=3)

    # Value labels
    for bar in bars_tr:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{bar.get_height():.1f}", ha="center", va="bottom",
                fontsize=6, color=DARK_GRAY)
    for bar in bars_te:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{bar.get_height():.1f}", ha="center", va="bottom",
                fontsize=6, color=DARK_GRAY)

    # Best test accuracy marker
    best_idx = int(np.argmax(tests))
    ax.annotate("best", xy=(x[best_idx] + width / 2, tests[best_idx]),
                xytext=(x[best_idx] + width / 2 + 0.25, tests[best_idx] + 2.5),
                fontsize=5.5, color=ORANGE,
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=0.65))

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(y_lo, 107)
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))
    ax.grid(axis="y", lw=0.4, alpha=0.5, zorder=0)
    ax.legend(loc="lower right", framealpha=0.85)
    ax.set_title(f"{title_tag}  {subtitle}",
                 loc="left", fontsize=8.5, pad=6, color=DARK_GRAY)

    out = os.path.join(OUT_DIR, out_name)
    fig.savefig(out, format="pdf")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    draw_benchmark(UTHAR,   "(a)", "UT-HAR (7-class, supervised)",
                   "bench_uthar.pdf",    y_lo=70.0)
    draw_benchmark(NTUFI,   "(b)", "NTU-Fi HAR (6-class, supervised)",
                   "bench_ntufi.pdf",    y_lo=75.0)
    draw_benchmark(HUMANID, "(c)", "NTU-Human ID (14-subject, SSL fine-tune)",
                   "bench_ntuhumanid.pdf", y_lo=45.0)
    print("Done.")
