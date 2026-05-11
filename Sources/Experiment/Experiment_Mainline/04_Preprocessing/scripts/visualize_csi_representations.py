#!/usr/bin/env python
"""
Visualize CSI data in three representations:
1. 1D Sequence (original temporal-spectral 2D)
2. 2D STFT (per-band spectro-temporal)
3. ACF (autocorrelation function)

This script demonstrates the preprocessing pipeline used in the WiSe4Car / XRF55
cross-domain transfer learning setup.
"""

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


# Paper-style plotting
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

EPS = 1e-6


def compute_stft_per_band(
    x_1d: np.ndarray,
    nperseg: int = 64,
    noverlap: int = 48,
    nfft: int = 128,
    use_power: bool = True,
    use_log1p: bool = True,
) -> np.ndarray:
    """
    Compute STFT for each frequency band independently.
    
    Input: x_1d [T, F] where T=256, F=16
    Output: spec [F, H, W] where H=(nfft//2+1), W=num_frames
    """
    t_len, n_bands = x_1d.shape
    per_band = []
    
    for b in range(n_bands):
        _, _, z = signal.stft(
            x_1d[:, b],
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            boundary=None,
            padded=False,
            return_onesided=True,
        )
        s = np.abs(z)
        if use_power:
            s = s ** 2
        if use_log1p:
            s = np.log1p(s)
        per_band.append(s.astype(np.float32))
    
    spec = np.stack(per_band, axis=0)
    mu = spec.mean()
    sigma = spec.std() + EPS
    spec = (spec - mu) / sigma
    spec = np.clip(spec, -6.0, 6.0)
    return spec.astype(np.float32)


def compute_acf_per_band(
    x_1d: np.ndarray,
    max_lag: int = 64,
    normalize_by_lag0: bool = True,
    standardize_per_feature: bool = True,
) -> np.ndarray:
    """
    Compute ACF for each frequency band independently.
    
    Input: x_1d [T, F] where T=256, F=16
    Output: acf [L, F] where L=max_lag+1=65, F=16
    """
    t_len, n_bands = x_1d.shape
    max_lag = min(max_lag, t_len - 1)
    
    acf_bands = []
    for b in range(n_bands):
        x0 = x_1d[:, b] - x_1d[:, b].mean()
        corr = np.correlate(x0, x0, mode="full")
        mid = len(corr) // 2
        acf = corr[mid : mid + max_lag + 1].astype(np.float32)
        if normalize_by_lag0:
            acf = acf / (float(acf[0]) + EPS)
        acf_bands.append(acf)
    
    acf_map = np.stack(acf_bands, axis=1)  # [L, F]
    
    if standardize_per_feature:
        mu = acf_map.mean(axis=0, keepdims=True)
        sigma = acf_map.std(axis=0, keepdims=True) + EPS
        acf_map = (acf_map - mu) / sigma
    
    acf_map = np.clip(acf_map, -6.0, 6.0)
    return acf_map.astype(np.float32)


def load_sample(domain: str = "xrf55", sample_idx: int = 0) -> Tuple[np.ndarray, str]:
    """
    Load a single sample from preprocessed data.
    
    Returns: (sample [256, 16], label_name)
    """
    data_root = Path(__file__).resolve().parents[3] / "Transfer_Learning" / "Data"
    
    if domain.lower() == "xrf55":
        features_path = data_root / "XRF55" / "XRF55_features.npy"
        labels_path = data_root / "XRF55" / "XRF55_labels.npy"
        names_path = data_root / "XRF55" / "XRF55_label_names.npy"
    elif domain.lower() == "wise4car":
        features_path = data_root / "Wise4Car" / "wise4car_unified_features.npy"
        labels_path = data_root / "Wise4Car" / "wise4car_unified_labels.npy"
        names_path = data_root / "Wise4Car" / "wise4car_unified_label_names.npy"
    else:
        raise ValueError(f"Unknown domain: {domain}")
    
    if not features_path.exists():
        raise FileNotFoundError(f"Data not found at {features_path}")
    
    features = np.load(features_path)
    labels = np.load(labels_path)
    label_names = np.load(names_path, allow_pickle=True)
    
    sample = features[sample_idx]
    label_id = labels[sample_idx]
    label_name = str(label_names[label_id])
    
    return sample, label_name


def visualize_representations(
    sample: np.ndarray,
    label_name: str,
    output_dir: Path,
) -> None:
    """Create a 3-panel figure showing all three representations."""
    
    # Compute all three representations
    seq_1d = sample  # [256, 16]
    stft_2d = compute_stft_per_band(seq_1d)  # [16, 65, 13]
    acf_2d = compute_acf_per_band(seq_1d)  # [65, 16]
    
    # Average across bands for visualization
    seq_avg = seq_1d.mean(axis=1)  # [256]
    stft_avg = stft_2d.mean(axis=0)  # [65, 13]
    acf_avg = acf_2d.mean(axis=1)  # [65]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    
    # Panel A: 1D Sequence (heatmap of all bands)
    ax_a = axes[0]
    im_a = ax_a.imshow(seq_1d.T, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    ax_a.set_title("(a) 1D CSI Sequence\n[T=256, F=16]", fontsize=11, fontweight="bold")
    ax_a.set_xlabel("Time steps")
    ax_a.set_ylabel("Frequency bands")
    ax_a.set_yticks([0, 8, 15])
    cbar_a = plt.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.04)
    cbar_a.set_label("Amplitude (z-score)", fontsize=9)
    
    # Panel B: STFT (averaged spectro-temporal map)
    ax_b = axes[1]
    im_b = ax_b.imshow(stft_avg, aspect="auto", cmap="viridis", vmin=-2, vmax=2)
    ax_b.set_title("(b) STFT (Avg. Across Bands)\n[F_freq=65, T_frames=13]", fontsize=11, fontweight="bold")
    ax_b.set_xlabel("STFT time frames")
    ax_b.set_ylabel("Frequency bins")
    cbar_b = plt.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)
    cbar_b.set_label("Log-Power (normalized)", fontsize=9)
    
    # Panel C: ACF (correlation lags across bands)
    ax_c = axes[2]
    im_c = ax_c.imshow(acf_2d, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax_c.set_title("(c) ACF (Per Band)\n[Lags=65, F=16]", fontsize=11, fontweight="bold")
    ax_c.set_xlabel("Frequency bands")
    ax_c.set_ylabel("Correlation lags")
    ax_c.set_xticks([0, 8, 15])
    cbar_c = plt.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.04)
    cbar_c.set_label("Normalized ACF", fontsize=9)
    
    # Overall title
    fig.suptitle(f"CSI Representation Comparison — Sample: {label_name}", 
                 fontsize=13, fontweight="bold", y=1.00)
    
    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"csi_representations_{label_name}.png"
    pdf_path = output_dir / f"csi_representations_{label_name}.pdf"
    
    fig.savefig(png_path, dpi=400, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    
    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")
    plt.close(fig)


def visualize_individual_bands(
    sample: np.ndarray,
    label_name: str,
    output_dir: Path,
    n_bands_show: int = 4,
) -> None:
    """Create a detailed figure showing STFT for selected bands."""
    
    stft_2d = compute_stft_per_band(sample)  # [16, 65, 13]
    
    # Select bands evenly spaced
    band_indices = np.linspace(0, 15, n_bands_show, dtype=int)
    
    fig, axes = plt.subplots(2, n_bands_show, figsize=(16, 6.5), constrained_layout=True)
    if n_bands_show == 1:
        axes = axes.reshape(2, 1)
    
    for i, b_idx in enumerate(band_indices):
        # Top row: STFT
        ax_top = axes[0, i]
        im = ax_top.imshow(stft_2d[b_idx], aspect="auto", cmap="viridis")
        ax_top.set_title(f"Band {b_idx}", fontsize=10)
        ax_top.set_ylabel("Freq. bins" if i == 0 else "")
        plt.colorbar(im, ax=ax_top, fraction=0.046, pad=0.04)
        
        # Bottom row: ACF of this band
        ax_bot = axes[1, i]
        acf_band = compute_acf_per_band(sample)[:, b_idx]
        ax_bot.plot(acf_band, linewidth=2.0, color="#1f77b4")
        ax_bot.fill_between(range(len(acf_band)), acf_band, alpha=0.3, color="#1f77b4")
        ax_bot.set_ylabel("ACF" if i == 0 else "")
        ax_bot.set_xlabel("Lag")
        ax_bot.grid(True, alpha=0.3)
        ax_bot.set_ylim(-1.2, 1.2)
    
    fig.suptitle(f"Per-Band STFT & ACF — Sample: {label_name}",
                 fontsize=13, fontweight="bold")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"per_band_analysis_{label_name}.png"
    pdf_path = output_dir / f"per_band_analysis_{label_name}.pdf"
    
    fig.savefig(png_path, dpi=400, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    
    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize CSI representations")
    parser.add_argument("--domain", choices=["xrf55", "wise4car"], default="xrf55",
                        help="Data domain to load")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Sample index to visualize")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Output directory for figures")
    parser.add_argument("--per_band", action="store_true",
                        help="Also generate per-band detailed analysis")
    
    args = parser.parse_args()
    
    try:
        sample, label_name = load_sample(domain=args.domain, sample_idx=args.sample_idx)
        print(f"Loaded {args.domain} sample {args.sample_idx}: {label_name}, shape {sample.shape}")
        
        output_path = Path(args.output_dir)
        
        # Main comparison figure
        visualize_representations(sample, label_name, output_path)
        
        # Optional per-band analysis
        if args.per_band:
            visualize_individual_bands(sample, label_name, output_path, n_bands_show=4)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
