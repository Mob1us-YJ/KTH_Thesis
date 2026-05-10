"""
Generate publication-quality ACF advantage figure (Fig. 5.4).
Compares 'turning' vs 'using phone' in two representations:
  (top row)    1D amplitude matrix   – raw temporal structure
  (bottom row) ACF (sub-band × lag)  – temporal self-similarity
The ACF discriminates the two activities more clearly than raw amplitude alone.
  - Turning:      slow broad modulation → broad positive correlation at long lags
  - Using phone:  ~5 Hz repetitive motion → periodic ACF spikes at 1/f_m ≈ 0.18 s
Output: figures/why_acf_turning.pdf
Run from thesis root: python figures/gen_acf_turning_figure.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import signal
import os

# ── Style (matches gen_repr_figure.py) ────────────────────────────────────────
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


# ── Signal simulation + preprocessing ─────────────────────────────────────────
def simulate_activity(activity, T=256, F=16, fs=50.0, seed=42):
    """
    Simulate a preprocessed CSI window for 'turning' or 'using_phone'.
    Turning: slow upper-body rotation (~0.5 Hz), single sustained sweep.
    Using phone: repetitive fine hand motion (~5 Hz dominant + harmonics).
    Applies the same five-stage pipeline as Section 3.3.
    """
    rng = np.random.RandomState(seed)
    t   = np.arange(T) / fs
    ev_s = int(T * 0.28)
    ev_e = int(T * 0.72)
    elen = ev_e - ev_s

    raw = np.zeros((T, F))
    for fi in range(F):
        base = (0.7 + 0.3 * np.sin(fi * 0.45 + 0.8)
                + 0.06 * np.sin(2 * np.pi * 0.08 * t + fi * 0.3)
                + 0.03 * t)
        noise = rng.randn(T) * 0.04

        env = np.zeros(T)
        env[ev_s:ev_e] = (
            np.sin(np.linspace(0, np.pi, elen)) ** 0.6
            * (0.45 + 0.10 * np.sin(fi * 0.55 + 1.1))
        )

        if activity == "turning":
            # Slow upper-body rotation: low frequency, single broad sweep
            fm = 0.45 + fi * 0.025
            act = env * (
                np.sin(2 * np.pi * fm * t + fi * 0.3)
                + 0.18 * np.sin(2 * np.pi * 2 * fm * t)
            )
        else:  # using_phone
            # Fine repetitive hand motion: ~5 Hz dominant + harmonics
            fm = 5.0 + fi * 0.09
            act = env * (
                np.sin(2 * np.pi * fm * t)
                + 0.28 * np.sin(2 * np.pi * 2 * fm * t)
                + 0.10 * np.sin(2 * np.pi * 3 * fm * t)
            )

        raw[:, fi] = base + noise + act

    # Stage 1 – Savitzky-Golay smoothing
    proc = signal.savgol_filter(raw, window_length=7, polyorder=3, axis=0)

    # Stage 2 – High-pass Butterworth (5th order, wc = 0.05 × Nyquist)
    sos  = signal.butter(5, 0.05, btype="high", output="sos")
    proc = signal.sosfilt(sos, proc, axis=0)

    # Stage 3 – Demean + linear detrend per sub-band
    proc -= proc.mean(axis=0)
    idx  = np.arange(T, dtype=float)
    for fi in range(F):
        p = np.polyfit(idx, proc[:, fi], 1)
        proc[:, fi] -= np.polyval(p, idx)

    # Stage 4 – Z-score normalisation
    proc = (proc - proc.mean(axis=0)) / (proc.std(axis=0) + 1e-8)
    return proc, t, ev_s, ev_e


# ── Representation computations ────────────────────────────────────────────────
def acf_matrix(data, max_lag=64):
    """Normalized biased ACF matrix (F × max_lag), lags 1..max_lag."""
    T, F = data.shape
    acf  = np.zeros((F, max_lag))
    for fi in range(F):
        x  = data[:, fi] - data[:, fi].mean()
        r0 = float(np.dot(x, x)) + 1e-12
        for lag in range(1, max_lag + 1):
            acf[fi, lag - 1] = float(np.dot(x[:T - lag], x[lag:])) / r0
    return acf


# ── Main figure ────────────────────────────────────────────────────────────────
def draw_acf_turning():
    fs = 50.0
    data_t, t, ev_s_t, ev_e_t = simulate_activity("turning",     seed=42)
    data_p, _, ev_s_p, ev_e_p = simulate_activity("using_phone", seed=7)

    acf_t = acf_matrix(data_t, max_lag=64)
    acf_p = acf_matrix(data_p, max_lag=64)
    lags  = np.arange(1, 65) / fs
    F     = data_t.shape[1]

    fig = plt.figure(figsize=(7.2, 4.4))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.54, wspace=0.46,
                            left=0.09, right=0.97, bottom=0.09, top=0.90)

    activities = [
        ("turning",     data_t, acf_t, ev_s_t, ev_e_t),
        ("using_phone", data_p, acf_p, ev_s_p, ev_e_p),
    ]
    col_labels = ["Turning", "Using phone"]

    for col, (act, data, acf, ev_s, ev_e) in enumerate(activities):
        tev_s, tev_e = ev_s / fs, ev_e / fs

        # ── top: amplitude heatmap ────────────────────────────────────────────
        ax_top = fig.add_subplot(gs[0, col])
        im_t   = ax_top.imshow(
            data.T, aspect="auto", origin="lower", cmap="Blues",
            interpolation="nearest",
            extent=[t[0], t[-1], 0.5, F + 0.5],
        )
        ax_top.set_xlim(t[0], t[-1])
        ax_top.set_ylim(0.5, F + 0.5)
        for xv in (tev_s, tev_e):
            ax_top.axvline(xv, color=ORANGE, lw=0.9, ls="--", alpha=0.85)
        # Bracket above
        ytop = F + 0.6
        ax_top.plot([tev_s, tev_e], [ytop, ytop],
                    color=ORANGE, lw=0.85, clip_on=False)
        ax_top.plot([tev_s, tev_s], [ytop - 0.25, ytop],
                    color=ORANGE, lw=0.85, clip_on=False)
        ax_top.plot([tev_e, tev_e], [ytop - 0.25, ytop],
                    color=ORANGE, lw=0.85, clip_on=False)
        ax_top.text((tev_s + tev_e) / 2, ytop + 0.2, "event",
                    ha="center", va="bottom", fontsize=6, color=ORANGE,
                    clip_on=False)
        ax_top.set_xlabel("Time (s)", labelpad=1)
        ax_top.set_ylabel("Sub-band", labelpad=1)
        letter = "a" if col == 0 else "b"
        ax_top.set_title(
            f"({letter}₁)  {col_labels[col]} — amplitude",
            loc="left", fontsize=8.5, pad=14, color=DARK_GRAY,
        )
        cb1 = fig.colorbar(im_t, ax=ax_top, fraction=0.046, pad=0.04)
        cb1.set_label("Norm. ampl.", fontsize=6.5)
        cb1.ax.tick_params(labelsize=6)

        # ── bottom: ACF heatmap ───────────────────────────────────────────────
        ax_bot = fig.add_subplot(gs[1, col])
        im_b   = ax_bot.imshow(
            acf, aspect="auto", origin="lower", cmap="RdBu_r",
            interpolation="nearest", vmin=-0.55, vmax=0.55,
            extent=[lags[0], lags[-1], 0.5, F + 0.5],
        )
        ax_bot.set_xlim(lags[0], lags[-1])
        ax_bot.set_ylim(0.5, F + 0.5)
        ax_bot.set_xlabel("Lag (s)", labelpad=1)
        ax_bot.set_ylabel("Sub-band", labelpad=1)
        ax_bot.set_title(
            f"({letter}₂)  {col_labels[col]} — ACF",
            loc="left", fontsize=8.5, pad=6, color=DARK_GRAY,
        )

        if col == 0:
            # Turning: highlight broad positive correlation region (slow motion)
            ax_bot.axvspan(lags[0], 0.75, alpha=0.12, color=KTH_BLUE, lw=0)
            ax_bot.text(0.37, 0.93, "broad corr.\n(slow rotation)",
                        ha="center", va="top", fontsize=5.5, color=KTH_BLUE,
                        style="italic", transform=ax_bot.transAxes)
        else:
            # Using phone: mark 1/f_motion periodic peak
            period = 1.0 / 5.5
            ax_bot.axvline(period, color=ORANGE, lw=0.9, ls="--", alpha=0.85)
            ax_bot.text(period + 0.013, 0.05, r"$1/f_m$",
                        ha="left", va="bottom", fontsize=6, color=ORANGE,
                        transform=ax_bot.get_xaxis_transform())

        cb2 = fig.colorbar(im_b, ax=ax_bot, fraction=0.046, pad=0.04)
        cb2.set_label("Correlation", fontsize=6.5)
        cb2.ax.tick_params(labelsize=6)

    out = os.path.join(OUT_DIR, "why_acf_turning.pdf")
    fig.savefig(out, format="pdf")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    draw_acf_turning()
    print("Done.")
