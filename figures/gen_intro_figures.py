"""
Generate publication-quality figures for Chapter 1 (Introduction).
Output: figures/fig_csi_principle.pdf
        figures/fig_application_taxonomy.pdf
        figures/fig_research_framework.pdf
Run from the thesis root: python figures/gen_intro_figures.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Arc
from matplotlib.lines import Line2D
import numpy as np
import os

# ── Shared style ────────────────────────────────────────────────────────────
KTH_BLUE   = "#1954a6"
KTH_LBLUE  = "#24a0d8"
DARK_GRAY  = "#3a3a3a"
MED_GRAY   = "#6e6e6e"
LIGHT_GRAY = "#d0d0d0"
ORANGE     = "#c44601"
GREEN      = "#008e4a"
OUT_DIR    = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        9,
    "axes.labelsize":   9,
    "axes.titlesize":   10,
    "legend.fontsize":  8,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "text.usetex":      False,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.05,
})

# ════════════════════════════════════════════════════════════════════════════
# Figure 1 – CSI Sensing Principle
# ════════════════════════════════════════════════════════════════════════════
def draw_csi_principle():
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8),
                             gridspec_kw={"width_ratios": [1.6, 1]})

    # ── Left panel: multipath scene ─────────────────────────────────────
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.axis("off")

    # Room walls
    room = mpatches.FancyBboxPatch((0.3, 0.3), 9.4, 5.4,
                                   boxstyle="square,pad=0",
                                   linewidth=1.5, edgecolor=DARK_GRAY,
                                   facecolor="#f7f7f7")
    ax.add_patch(room)

    # Ceiling label
    ax.text(5, 5.75, "Room / Vehicle Cabin", ha="center", va="center",
            fontsize=7.5, color=MED_GRAY, style="italic")

    # TX (access point)
    tx_x, tx_y = 1.2, 3.0
    tx_box = FancyBboxPatch((tx_x-0.45, tx_y-0.55), 0.9, 1.1,
                             boxstyle="round,pad=0.05",
                             facecolor=KTH_BLUE, edgecolor="none", zorder=3)
    ax.add_patch(tx_box)
    ax.text(tx_x, tx_y+0.05, "TX", ha="center", va="center",
            fontsize=8, color="white", fontweight="bold", zorder=4)
    ax.text(tx_x, tx_y-0.8, "Wi-Fi AP", ha="center", va="top",
            fontsize=7, color=KTH_BLUE)

    # RX (client device)
    rx_x, rx_y = 8.8, 3.0
    rx_box = FancyBboxPatch((rx_x-0.45, rx_y-0.55), 0.9, 1.1,
                             boxstyle="round,pad=0.05",
                             facecolor=KTH_LBLUE, edgecolor="none", zorder=3)
    ax.add_patch(rx_box)
    ax.text(rx_x, rx_y+0.05, "RX", ha="center", va="center",
            fontsize=8, color="white", fontweight="bold", zorder=4)
    ax.text(rx_x, rx_y-0.8, "Wi-Fi Client", ha="center", va="top",
            fontsize=7, color=KTH_LBLUE)

    # Human silhouette (simplified circles + rectangle)
    hx, hy = 5.0, 2.2
    head = plt.Circle((hx, hy+1.65), 0.3, color=ORANGE, zorder=3)
    body = mpatches.FancyBboxPatch((hx-0.28, hy+0.5), 0.56, 1.0,
                                    boxstyle="round,pad=0.05",
                                    facecolor=ORANGE, edgecolor="none", zorder=3)
    ax.add_patch(head)
    ax.add_patch(body)
    # legs
    ax.plot([hx-0.15, hx-0.25, hx-0.3], [hy+0.5, hy+0.1, hy-0.3],
            color=ORANGE, lw=2, zorder=3)
    ax.plot([hx+0.15, hx+0.25, hx+0.3], [hy+0.5, hy+0.1, hy-0.3],
            color=ORANGE, lw=2, zorder=3)
    # arms
    ax.plot([hx-0.28, hx-0.55, hx-0.5], [hy+1.3, hy+0.9, hy+0.5],
            color=ORANGE, lw=2, zorder=3)
    ax.plot([hx+0.28, hx+0.55, hx+0.5], [hy+1.3, hy+0.9, hy+0.5],
            color=ORANGE, lw=2, zorder=3)
    ax.text(hx, hy-0.6, "Occupant", ha="center", va="top",
            fontsize=7, color=ORANGE)

    # Direct path (dashed grey)
    ax.annotate("", xy=(rx_x-0.45, rx_y), xytext=(tx_x+0.45, tx_y),
                arrowprops=dict(arrowstyle="-|>", color=MED_GRAY,
                                lw=1.2, linestyle="dashed"))
    ax.text(5.0, 3.35, "Direct path", ha="center", va="bottom",
            fontsize=6.5, color=MED_GRAY, style="italic")

    # Reflected path via ceiling
    ax.annotate("", xy=(rx_x-0.45, rx_y+0.3), xytext=(5.0, 5.3),
                arrowprops=dict(arrowstyle="-|>", color=KTH_BLUE,
                                lw=1.0, connectionstyle="arc3,rad=0.0"))
    ax.annotate("", xy=(5.0, 5.3), xytext=(tx_x+0.45, tx_y+0.3),
                arrowprops=dict(arrowstyle="-|>", color=KTH_BLUE,
                                lw=1.0, connectionstyle="arc3,rad=0.0"))

    # Reflected path via floor
    ax.annotate("", xy=(rx_x-0.45, rx_y-0.3), xytext=(5.0, 0.7),
                arrowprops=dict(arrowstyle="-|>", color=KTH_BLUE,
                                lw=1.0, connectionstyle="arc3,rad=0.0"))
    ax.annotate("", xy=(5.0, 0.7), xytext=(tx_x+0.45, tx_y-0.3),
                arrowprops=dict(arrowstyle="-|>", color=KTH_BLUE,
                                lw=1.0, connectionstyle="arc3,rad=0.0"))

    # Body-scattered paths (orange, curved)
    ax.annotate("", xy=(rx_x-0.45, rx_y+0.1),
                xytext=(hx+0.35, hy+1.2),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE,
                                lw=1.2, connectionstyle="arc3,rad=-0.3"))
    ax.annotate("", xy=(hx+0.35, hy+1.0),
                xytext=(tx_x+0.45, tx_y+0.1),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE,
                                lw=1.2, connectionstyle="arc3,rad=-0.3"))

    # Legend
    legend_elems = [
        Line2D([0],[0], color=MED_GRAY,  lw=1.2, ls="--", label="Direct"),
        Line2D([0],[0], color=KTH_BLUE,  lw=1.0,         label="Wall-reflected"),
        Line2D([0],[0], color=ORANGE,    lw=1.2,         label="Body-scattered"),
    ]
    ax.legend(handles=legend_elems, loc="lower center",
              ncol=3, fontsize=6.5, framealpha=0.9,
              bbox_to_anchor=(0.5, -0.02))

    ax.set_title("(a) Multipath propagation perturbed by occupant",
                 fontsize=8.5, pad=4)

    # ── Right panel: CSI amplitude heatmap ──────────────────────────────
    ax2 = axes[1]
    np.random.seed(42)
    T, S = 80, 30
    # Simulate CSI: baseline + activity bump
    t = np.linspace(0, 4*np.pi, T)
    s = np.arange(S)
    base = np.outer(np.sin(s * 0.3) * 0.5 + 2.0,
                    np.ones(T))
    activity = np.zeros((S, T))
    for i, si in enumerate(s):
        freq = 0.8 + si * 0.05
        activity[i] = np.exp(-((t - np.pi*2)**2) / 3.0) * \
                       np.sin(freq * t) * (0.4 + 0.02*si)
    noise = np.random.randn(S, T) * 0.08
    csi_amp = base + activity + noise

    im = ax2.imshow(csi_amp, aspect="auto", cmap="viridis",
                    origin="lower", extent=[0, T, 0, S])
    plt.colorbar(im, ax=ax2, label="Amplitude (norm.)", fraction=0.046, pad=0.04)
    ax2.set_xlabel("Time (frames)")
    ax2.set_ylabel("Subcarrier index")
    ax2.set_title("(b) CSI amplitude matrix\n(activity perturbation visible)",
                  fontsize=8.5, pad=4)

    # Annotate perturbation region
    ax2.annotate("Motion\npattern",
                 xy=(45, 15), xytext=(58, 8),
                 fontsize=6.5, color="white",
                 arrowprops=dict(arrowstyle="->", color="white", lw=0.8))

    plt.tight_layout(w_pad=0.5)
    out = os.path.join(OUT_DIR, "fig_csi_principle.pdf")
    fig.savefig(out, format="pdf")
    plt.close(fig)
    print(f"Saved {out}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 2 – Wi-Fi Sensing Application Taxonomy
# ════════════════════════════════════════════════════════════════════════════
def draw_application_taxonomy():
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.8)
    ax.axis("off")

    # ── Category columns ────────────────────────────────────────────────
    categories = [
        {"label": "Presence &\nLocalization", "x": 1.15, "color": KTH_BLUE,
         "items": ["Occupancy\ndetection", "Indoor\nlocalization",
                   "Person\ncounting"]},
        {"label": "Activity\nRecognition", "x": 3.45, "color": KTH_LBLUE,
         "items": ["HAR\n(walking, sitting…)", "Gesture\nrecognition",
                   "Pose\nestimation"]},
        {"label": "Vital Signs\nMonitoring", "x": 5.75, "color": GREEN,
         "items": ["Breathing\nrate", "Heart rate\n& HRV",
                   "Sleep\nmonitoring"]},
        {"label": "Safety &\nAutomotive", "x": 8.05, "color": ORANGE,
         "items": ["Child presence\ndetection", "Driver fatigue\nmonitoring",
                   "In-cabin activity\nrecognition"]},
    ]

    col_w = 1.8
    for cat in categories:
        cx = cat["x"]
        col = cat["color"]
        # Header box
        hdr = FancyBboxPatch((cx - col_w/2, 4.35), col_w, 1.1,
                              boxstyle="round,pad=0.08",
                              facecolor=col, edgecolor="none", zorder=2)
        ax.add_patch(hdr)
        ax.text(cx, 4.9, cat["label"], ha="center", va="center",
                fontsize=8, color="white", fontweight="bold",
                multialignment="center", zorder=3)

        # Item boxes
        for j, item in enumerate(cat["items"]):
            iy = 3.2 - j * 1.15
            item_box = FancyBboxPatch((cx - col_w/2 + 0.06, iy - 0.42),
                                       col_w - 0.12, 0.84,
                                       boxstyle="round,pad=0.06",
                                       facecolor="white",
                                       edgecolor=col, linewidth=0.8, zorder=2)
            ax.add_patch(item_box)
            ax.text(cx, iy, item, ha="center", va="center",
                    fontsize=7, color=DARK_GRAY,
                    multialignment="center", zorder=3)
            # Connector from header to first item
            if j == 0:
                ax.plot([cx, cx], [4.35, iy + 0.42],
                        color=col, lw=0.8, zorder=1)
            else:
                ax.plot([cx, cx], [iy + 1.15 - 0.42, iy + 0.42],
                        color=col, lw=0.8, zorder=1)

    # ── Top banner ───────────────────────────────────────────────────────
    banner = FancyBboxPatch((0.2, 5.3), 9.6, 0.42,
                             boxstyle="round,pad=0.05",
                             facecolor=DARK_GRAY, edgecolor="none")
    ax.add_patch(banner)
    ax.text(5.0, 5.51, "Wi-Fi CSI Sensing — Application Landscape",
            ha="center", va="center", fontsize=9,
            color="white", fontweight="bold")

    # ── This thesis focus marker ──────────────────────────────────────────
    focus_box = FancyBboxPatch((7.1, -0.15), 1.9, 4.15,
                                boxstyle="round,pad=0.1",
                                facecolor="none", edgecolor=ORANGE,
                                linewidth=1.8, linestyle="--", zorder=0)
    ax.add_patch(focus_box)
    ax.text(8.05, -0.3, "This thesis", ha="center", va="top",
            fontsize=7.5, color=ORANGE, fontweight="bold")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig_application_taxonomy.pdf")
    fig.savefig(out, format="pdf")
    plt.close(fig)
    print(f"Saved {out}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 3 – Research Framework Overview
# ════════════════════════════════════════════════════════════════════════════
def draw_research_framework():
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # ── Helper: rounded box ───────────────────────────────────────────────
    def rbox(ax, x, y, w, h, color, text, fontsize=7.8, textcolor="white",
             linestyle="solid", facecolor=None, edgecolor=None):
        fc = facecolor if facecolor else color
        ec = edgecolor if edgecolor else color
        b = FancyBboxPatch((x - w/2, y - h/2), w, h,
                            boxstyle="round,pad=0.1",
                            facecolor=fc, edgecolor=ec,
                            linewidth=1.2, linestyle=linestyle, zorder=2)
        ax.add_patch(b)
        ax.text(x, y, text, ha="center", va="center",
                fontsize=fontsize, color=textcolor,
                multialignment="center", zorder=3)

    def arrow(ax, x1, y1, x2, y2, label="", color=DARK_GRAY):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=1.3, mutation_scale=10),
                    zorder=1)
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx, my+0.15, label, ha="center", va="bottom",
                    fontsize=6.5, color=color, style="italic")

    # ── Row labels ───────────────────────────────────────────────────────
    ax.text(0.15, 4.3, "Datasets", ha="left", va="center",
            fontsize=7.5, color=MED_GRAY, style="italic")
    ax.text(0.15, 2.5, "Stage", ha="left", va="center",
            fontsize=7.5, color=MED_GRAY, style="italic")
    ax.text(0.15, 0.7, "Output", ha="left", va="center",
            fontsize=7.5, color=MED_GRAY, style="italic")

    # ── Source datasets (top row) ─────────────────────────────────────────
    src_datasets = [
        ("UT-HAR\n(7 cls)", 2.0),
        ("NTU-Fi HAR\n(6 cls)", 3.8),
        ("Widar3.0\n(22 cls)", 5.6),
        ("XRF55\n(55 cls)", 7.4),
    ]
    for name, dx in src_datasets:
        rbox(ax, dx, 4.3, 1.5, 0.7, KTH_BLUE, name, fontsize=7.0)

    # WiSe4Car dataset (separate, automotive)
    rbox(ax, 10.5, 4.3, 2.0, 0.7, ORANGE, "WiSe4Car\n(in-vehicle)", fontsize=7.0)

    # ── Stage 1: Benchmark ────────────────────────────────────────────────
    rbox(ax, 3.8, 2.5, 3.6, 0.85, KTH_BLUE,
         "Stage 1 — Indoor Benchmark\n1D-CNN / 2D-CNN / CNN+GRU / Transformer")
    for _, dx in src_datasets[:3]:
        arrow(ax, dx, 3.95, 3.8, 2.95)

    # ── Stage 2: Pre-training ─────────────────────────────────────────────
    rbox(ax, 7.4, 2.5, 2.0, 0.85, KTH_BLUE,
         "Stage 2\nSource Pre-train\n(XRF55)")
    arrow(ax, 7.4, 3.95, 7.4, 2.95)

    # ── SSL Adaptation ───────────────────────────────────────────────────
    rbox(ax, 10.5, 2.5, 2.0, 0.85, "#c44601",
         "Stage 3\nSSL Domain\nAdaptation")
    arrow(ax, 7.4, 2.5, 9.5, 2.5, "transfer")
    arrow(ax, 10.5, 3.95, 10.5, 2.95)

    # ── Outputs (bottom row) ─────────────────────────────────────────────
    rbox(ax, 3.8, 0.7, 3.4, 0.75, "white",
         "Benchmark accuracy\n& architecture comparison",
         textcolor=KTH_BLUE, facecolor="white", edgecolor=KTH_BLUE,
         linestyle="dashed")
    arrow(ax, 3.8, 2.08, 3.8, 1.08)

    rbox(ax, 10.5, 0.7, 2.4, 0.75, "white",
         "In-cabin HAR accuracy\n(RQ1 answer)",
         textcolor=ORANGE, facecolor="white", edgecolor=ORANGE,
         linestyle="dashed")
    arrow(ax, 10.5, 2.08, 10.5, 1.08)

    # Few-shot fine-tune connector
    arrow(ax, 10.5, 2.08, 10.5, 1.08)
    ax.text(10.5, 1.55, "Few-shot\nfine-tune →", ha="center", va="center",
            fontsize=6.5, color=ORANGE, style="italic")

    # ── RQ labels ─────────────────────────────────────────────────────────
    ax.text(3.8, -0.05, "→ RQ2: Design choices", ha="center", va="top",
            fontsize=7, color=KTH_BLUE, fontweight="bold")
    ax.text(10.5, -0.05, "→ RQ1: Feasibility", ha="center", va="top",
            fontsize=7, color=ORANGE, fontweight="bold")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig_research_framework.pdf")
    fig.savefig(out, format="pdf")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    draw_csi_principle()
    draw_application_taxonomy()
    draw_research_framework()
    print("All figures generated.")
