"""
Generate publication-quality CSI representation comparison figure (Fig. 5.3).
Shows the same simulated 'using phone' WiSe4Car window in three representations:
  (a) 1D amplitude matrix   – raw temporal structure
  (b) STFT spectrogram      – time-frequency content
  (c) ACF (sub-band × lag)  – temporal self-similarity / periodicity
Output: figures/csi_representations.pdf
Run from thesis root: python figures/gen_repr_figure.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import signal
import os

# ── Style (matches gen_intro_figures.py) ─────────────────────────────────────
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


# ── Signal simulation + preprocessing ────────────────────────────────────────
def simulate_and_preprocess(T=256, F=16, fs=50.0, seed=42):
    """
    Simulate a 'using phone' CSI window and apply the five-stage preprocessing
    pipeline from Section 3.3 (SavGol, HP filter, demean/detrend, z-score).
    Returns: data (T, F), time vector t (T,), event start/end sample indices.
    """
    rng = np.random.RandomState(seed)
    t   = np.arange(T) / fs
    ev_s = int(T * 0.28)           # event onset  ~1.4 s
    ev_e = int(T * 0.68)           # event offset ~3.5 s
    elen = ev_e - ev_s

    raw = np.zeros((T, F))
    for fi in range(F):
        # Slowly-varying baseline + slight linear drift
        base = (0.7 + 0.3 * np.sin(fi * 0.45 + 0.8)
                + 0.06 * np.sin(2 * np.pi * 0.08 * t + fi * 0.3)
                + 0.04 * t)
        # Sensor noise
        noise = rng.randn(T) * 0.05

        # Smooth event envelope
        env = np.zeros(T)
        env[ev_s:ev_e] = (
            np.sin(np.linspace(0, np.pi, elen)) ** 0.75
            * (0.42 + 0.12 * np.sin(fi * 0.55 + 1.1))
        )
        # Phone-interaction motion: ~5 Hz dominant + harmonics,
        # frequency varies slightly across sub-bands (realistic spread)
        fm = 5.0 + fi * 0.09
        act = env * (
            np.sin(2 * np.pi * fm * t)
            + 0.28 * np.sin(2 * np.pi * 2 * fm * t)
            + 0.10 * np.sin(2 * np.pi * 3 * fm * t)
        )
        raw[:, fi] = base + noise + act

    # Step 1 – Savitzky-Golay smoothing (matches Section 3.3 step 3)
    proc = signal.savgol_filter(raw, window_length=7, polyorder=3, axis=0)

    # Step 2 – High-pass Butterworth (5th order, wc = 0.05 × Nyquist)
    sos  = signal.butter(5, 0.05, btype="high", output="sos")
    proc = signal.sosfilt(sos, proc, axis=0)

    # Step 3 – Demean + linear detrend per sub-band
    proc -= proc.mean(axis=0)
    idx  = np.arange(T, dtype=float)
    for fi in range(F):
        p          = np.polyfit(idx, proc[:, fi], 1)
        proc[:, fi] -= np.polyval(p, idx)

    # Step 4 – Z-score normalisation
    proc = (proc - proc.mean(axis=0)) / (proc.std(axis=0) + 1e-8)
    return proc, t, ev_s, ev_e


# ── Representation computations ───────────────────────────────────────────────
def mean_stft(data, fs=50.0, nperseg=64, noverlap=48):
    """
    Mean log-power STFT across all F sub-bands.
    Parameters match Section 3.3.4 (nperseg=64, noverlap=48, nfft=64).
    """
    freqs, t_stft, Z0 = signal.stft(data[:, 0], fs=fs,
                                     nperseg=nperseg, noverlap=noverlap,
                                     nfft=nperseg)
    logpow = np.zeros_like(np.abs(Z0))
    for fi in range(data.shape[1]):
        _, _, Z  = signal.stft(data[:, fi], fs=fs,
                               nperseg=nperseg, noverlap=noverlap,
                               nfft=nperseg)
        logpow  += np.log1p(np.abs(Z))
    return freqs, t_stft, logpow / data.shape[1]


def acf_matrix(data, max_lag=64):
    """
    Normalized biased ACF matrix (F × max_lag), lags 1..max_lag.
    Matches the ACF representation in Section 3.3.4.
    """
    T, F = data.shape
    acf  = np.zeros((F, max_lag))
    for fi in range(F):
        x  = data[:, fi] - data[:, fi].mean()
        r0 = float(np.dot(x, x)) + 1e-12
        for lag in range(1, max_lag + 1):
            acf[fi, lag - 1] = float(np.dot(x[: T - lag], x[lag:])) / r0
    return acf


# ── Main figure ───────────────────────────────────────────────────────────────
def draw_csi_representations():
    data, t, ev_s, ev_e = simulate_and_preprocess()
    T, F = data.shape
    fs   = 50.0
    tev_s, tev_e = ev_s / fs, ev_e / fs

    freqs, t_stft, logpow = mean_stft(data, fs)
    acf  = acf_matrix(data, max_lag=64)
    lags = np.arange(1, 65) / fs           # lags in seconds

    fig = plt.figure(figsize=(7.2, 2.75))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.55,
                            left=0.07, right=0.97, bottom=0.17, top=0.82)

    # ── (a) 1D amplitude matrix ───────────────────────────────────────────────
    ax1  = fig.add_subplot(gs[0])
    im1  = ax1.imshow(
        data.T, aspect="auto", origin="lower", cmap="Blues",
        interpolation="nearest", extent=[t[0], t[-1], 0.5, F + 0.5],
    )
    ax1.set_xlim(t[0], t[-1])
    ax1.set_ylim(0.5, F + 0.5)
    ax1.set_xlabel("Time (s)", labelpad=1)
    ax1.set_ylabel("Sub-band", labelpad=1)

    # Event region: dashed verticals
    for xv in (tev_s, tev_e):
        ax1.axvline(xv, color=ORANGE, lw=0.9, ls="--", alpha=0.85,
                    clip_on=False)
    # Bracket above axes
    ytop = F + 0.6
    ax1.plot([tev_s, tev_e], [ytop, ytop],
             color=ORANGE, lw=0.85, clip_on=False)
    ax1.plot([tev_s, tev_s], [ytop - 0.25, ytop],
             color=ORANGE, lw=0.85, clip_on=False)
    ax1.plot([tev_e, tev_e], [ytop - 0.25, ytop],
             color=ORANGE, lw=0.85, clip_on=False)
    ax1.text((tev_s + tev_e) / 2, ytop + 0.2, "event",
             ha="center", va="bottom", fontsize=6, color=ORANGE,
             clip_on=False)

    ax1.set_title("(a)  1D amplitude matrix", loc="left",
                  fontsize=8.5, pad=14, color=DARK_GRAY)
    cb1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cb1.set_label("Norm. ampl.", fontsize=6.5)
    cb1.ax.tick_params(labelsize=6)

    # ── (b) STFT spectrogram (mean across sub-bands) ──────────────────────────
    ax2   = fig.add_subplot(gs[1])
    fmask = freqs <= 12.0
    im2   = ax2.imshow(
        logpow[fmask], aspect="auto", origin="lower",
        cmap="Blues", interpolation="bilinear",
        extent=[t_stft[0], t_stft[-1], freqs[0], freqs[fmask].max()],
    )
    for xv in (tev_s, tev_e):
        ax2.axvline(xv, color=ORANGE, lw=0.9, ls="--", alpha=0.85)
    # Highlight motion-frequency band (4–8 Hz)
    ax2.axhspan(4.0, 8.0, alpha=0.15, color=ORANGE, lw=0)
    ax2.text(0.97, 6.0, "4–8 Hz\nmotion",
             ha="right", va="center", fontsize=5.8, color=ORANGE,
             style="italic", transform=ax2.get_yaxis_transform())
    ax2.set_xlabel("Time (s)", labelpad=1)
    ax2.set_ylabel("Frequency (Hz)", labelpad=1)
    ax2.set_title("(b)  STFT spectrogram", loc="left",
                  fontsize=8.5, pad=14, color=DARK_GRAY)
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.set_label("log(1+|S|)", fontsize=6.5)
    cb2.ax.tick_params(labelsize=6)

    # ── (c) ACF  sub-band × lag ────────────────────────────────────────────────
    ax3  = fig.add_subplot(gs[2])
    im3  = ax3.imshow(
        acf, aspect="auto", origin="lower", cmap="RdBu_r",
        interpolation="nearest", vmin=-0.55, vmax=0.55,
        extent=[lags[0], lags[-1], 0.5, F + 0.5],
    )
    ax3.set_xlim(lags[0], lags[-1])
    ax3.set_ylim(0.5, F + 0.5)
    ax3.set_xlabel("Lag (s)", labelpad=1)
    ax3.set_ylabel("Sub-band", labelpad=1)

    # Mark 1/f_motion (period of dominant phone-motion frequency ~5.5 Hz)
    period = 1.0 / 5.5
    ax3.axvline(period, color=ORANGE, lw=0.9, ls="--", alpha=0.85)
    ax3.text(period + 0.012, 0.05, r"$1/f_m$",
             ha="left", va="bottom", fontsize=6, color=ORANGE,
             transform=ax3.get_xaxis_transform())

    ax3.set_title("(c)  ACF (sub-band × lag)", loc="left",
                  fontsize=8.5, pad=14, color=DARK_GRAY)
    cb3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cb3.set_label("Correlation", fontsize=6.5)
    cb3.ax.tick_params(labelsize=6)

    out = os.path.join(OUT_DIR, "csi_representations.pdf")
    fig.savefig(out, format="pdf")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    draw_csi_representations()
    print("Done.")
