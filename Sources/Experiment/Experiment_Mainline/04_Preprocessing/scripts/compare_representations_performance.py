#!/usr/bin/env python
"""
Performance Analysis: Compare why ACF > STFT > 1D

This script visualizes samples from a periodic action (e.g., Waving) across
all three representations to explain why ACF achieves better accuracy.

Key insight:
- 1D: Contains all info but mixed time-frequency
- STFT: Separates spectrum over time, but energy dispersed across bins
- ACF: Highlights temporal periodicity with sharp correlation peaks
"""

from pathlib import Path
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import signal


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

EPS = 1e-6


def load_representations(domain: str = "xrf55", class_name: str = "Waving", n_samples: int = 3):
    """Load multiple samples of a specific class across all representations."""
    data_root = Path(__file__).resolve().parents[3] / "Transfer_Learning" / "Data"
    
    if domain.lower() == "xrf55":
        prefix = "xrf55"
    elif domain.lower() == "wise4car":
        prefix = "wise4car"
    else:
        raise ValueError(f"Unknown domain: {domain}")
    
    # Load 1D data
    features_1d = np.load(data_root / "XRF55" / f"{prefix}_features.npy")
    labels_1d = np.load(data_root / "XRF55" / f"{prefix}_labels.npy")
    names_1d = np.load(data_root / "XRF55" / f"{prefix}_label_names.npy", allow_pickle=True)
    
    # Load STFT data
    features_stft = np.load(data_root / "STFT" / f"{prefix}_stft_features.npy")
    labels_stft = np.load(data_root / "STFT" / f"{prefix}_stft_labels.npy")
    names_stft = np.load(data_root / "STFT" / f"{prefix}_stft_label_names.npy", allow_pickle=True)
    
    # Load ACF data
    features_acf = np.load(data_root / "ACF" / f"{prefix}_acf_features.npy")
    labels_acf = np.load(data_root / "ACF" / f"{prefix}_acf_labels.npy")
    names_acf = np.load(data_root / "ACF" / f"{prefix}_acf_label_names.npy", allow_pickle=True)
    
    # Find class index
    class_idx = None
    for i, name in enumerate(names_1d):
        if str(name).lower() == class_name.lower():
            class_idx = i
            break
    
    if class_idx is None:
        raise ValueError(f"Class '{class_name}' not found. Available: {names_1d}")
    
    # Get sample indices for this class
    class_mask_1d = labels_1d == class_idx
    sample_indices_1d = np.where(class_mask_1d)[0][:n_samples]
    
    samples = []
    for idx in sample_indices_1d:
        # Handle ACF shape: (1, 65, 16) -> (65, 16)
        acf_sample = features_acf[idx].squeeze() if features_acf[idx].ndim > 2 else features_acf[idx]
        samples.append({
            "1d": features_1d[idx],
            "stft": features_stft[idx],
            "acf": acf_sample,
            "label": str(names_1d[class_idx])
        })
    
    return samples


def compute_statistical_features(x_1d: np.ndarray) -> dict:
    """Extract interpretable statistical features from 1D representation."""
    # Temporal statistics
    temporal_energy = np.mean(x_1d ** 2, axis=1)  # Energy per timestep [256]
    freq_energy = np.mean(x_1d ** 2, axis=0)     # Energy per frequency band [16]
    
    # Variability
    temporal_var = np.var(temporal_energy)
    freq_var = np.var(freq_energy)
    
    return {
        "temporal_energy": temporal_energy,
        "freq_energy": freq_energy,
        "temporal_var": temporal_var,
        "freq_var": freq_var,
        "mean_energy": np.mean(x_1d ** 2)
    }


def analyze_acf_periodicity(x_1d: np.ndarray, max_lag: int = 64):
    """Compute ACF and detect dominant periodicity."""
    # Compute per-band ACF (simplified version)
    acf_features = []
    dominant_lags = []
    
    for b in range(x_1d.shape[1]):
        x0 = x_1d[:, b] - x_1d[:, b].mean()
        corr = np.correlate(x0, x0, mode="full")
        mid = len(corr) // 2
        acf = corr[mid : mid + max_lag + 1]
        acf = acf / (float(acf[0]) + EPS)
        acf_features.append(acf)
        
        # Find dominant peak (exclude lag 0)
        if len(acf) > 1:
            peaks, _ = signal.find_peaks(acf[1:], height=0.3)
            if len(peaks) > 0:
                dominant_lag = peaks[0] + 1
                dominant_lags.append(dominant_lag)
    
    acf_map = np.stack(acf_features, axis=1)  # [65, 16]
    
    if dominant_lags:
        mean_period = np.mean(dominant_lags)
    else:
        mean_period = None
    
    return acf_map, mean_period, np.array(dominant_lags) if dominant_lags else None


def create_comparison_figure(samples, output_dir: Path):
    """Create a multi-panel figure comparing representations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    n_samples = len(samples)
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(18, 4 * n_samples + 1), constrained_layout=True)
    gs = GridSpec(n_samples, 5, figure=fig, hspace=0.4, wspace=0.35)
    
    label_name = samples[0]["label"]
    
    for sample_idx, sample in enumerate(samples):
        x_1d = sample["1d"]        # [256, 16]
        x_stft = sample["stft"]    # [16, 65, 13]
        x_acf = sample["acf"]      # [65, 16]
        
        row_offset = sample_idx
        
        # ============ Panel A: 1D heatmap ============
        ax_1d = fig.add_subplot(gs[row_offset, 0])
        im_1d = ax_1d.imshow(x_1d.T, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
        ax_1d.set_title(f"(a) 1D [{sample_idx+1}]", fontsize=10, fontweight="bold")
        ax_1d.set_ylabel("Freq band")
        if sample_idx == n_samples - 1:
            ax_1d.set_xlabel("Time steps")
        ax_1d.set_yticks([0, 8, 15])
        
        # ============ Panel B: STFT average across bands ============
        ax_stft = fig.add_subplot(gs[row_offset, 1])
        stft_avg = x_stft.mean(axis=0)  # Average across 16 bands
        im_stft = ax_stft.imshow(stft_avg, aspect="auto", cmap="viridis", vmin=-2, vmax=2)
        ax_stft.set_title(f"(b) STFT Avg [{sample_idx+1}]", fontsize=10, fontweight="bold")
        if sample_idx == n_samples - 1:
            ax_stft.set_xlabel("Time frames")
        ax_stft.set_ylabel("Freq bins")
        
        # ============ Panel C: ACF heatmap ============
        ax_acf = fig.add_subplot(gs[row_offset, 2])
        im_acf = ax_acf.imshow(x_acf, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
        ax_acf.set_title(f"(c) ACF [{sample_idx+1}]", fontsize=10, fontweight="bold")
        if sample_idx == n_samples - 1:
            ax_acf.set_xlabel("Freq band")
        ax_acf.set_ylabel("Lag")
        ax_acf.set_xticks([0, 8, 15])
        
        # ============ Panel D: Temporal energy curve ============
        ax_energy = fig.add_subplot(gs[row_offset, 3])
        stats = compute_statistical_features(x_1d)
        temporal_energy = stats["temporal_energy"]
        ax_energy.plot(temporal_energy, linewidth=2, color="#1f77b4", label="1D Energy")
        ax_energy.fill_between(range(len(temporal_energy)), temporal_energy, alpha=0.3, color="#1f77b4")
        ax_energy.set_title(f"(d) Temporal Energy [{sample_idx+1}]", fontsize=10, fontweight="bold")
        ax_energy.set_ylabel("Energy")
        if sample_idx == n_samples - 1:
            ax_energy.set_xlabel("Time steps")
        ax_energy.grid(True, alpha=0.3)
        ax_energy.set_ylim(bottom=0)
        
        # ============ Panel E: ACF periodicity detection ============
        ax_period = fig.add_subplot(gs[row_offset, 4])
        acf_map, mean_period, dom_lags = analyze_acf_periodicity(x_1d)
        # Show average ACF across bands
        acf_avg = acf_map.mean(axis=1)
        ax_period.plot(acf_avg, linewidth=2.5, color="#d62728", label="ACF (avg)")
        ax_period.fill_between(range(len(acf_avg)), acf_avg, alpha=0.3, color="#d62728")
        
        # Mark peaks
        peaks, _ = signal.find_peaks(acf_avg[1:], height=0.2)
        if len(peaks) > 0:
            peak_lags = peaks + 1
            ax_period.plot(peak_lags, acf_avg[peak_lags], 'go', markersize=8, 
                          label=f"Peaks (period~{peak_lags[0] if len(peak_lags) > 0 else 'N/A'})")
        
        ax_period.set_title(f"(e) ACF Periodicity [{sample_idx+1}]", fontsize=10, fontweight="bold")
        ax_period.set_ylabel("Correlation")
        if sample_idx == n_samples - 1:
            ax_period.set_xlabel("Lag")
        ax_period.set_ylim(-1.2, 1.2)
        ax_period.grid(True, alpha=0.3)
        ax_period.legend(fontsize=8, loc="upper right")
        
        # Print period analysis
        if mean_period is not None:
            print(f"Sample {sample_idx+1}: Detected period ≈ {mean_period:.1f} samples (~{mean_period/256*4.6:.2f}s)")
    
    # Overall title
    fig.suptitle(f"Why ACF > STFT > 1D — Class: {label_name}\n" +
                 "(a) 1D: Mixed time-freq | (b) STFT: Time-freq separated | " +
                 "(c) ACF: Temporal structure | (d) Energy: Temporal dynamics | " +
                 "(e) Periodicity: Key discriminator",
                 fontsize=12, fontweight="bold", y=0.995)
    
    # Save
    png_path = output_dir / f"why_acf_best_{label_name}.png"
    pdf_path = output_dir / f"why_acf_best_{label_name}.pdf"
    
    fig.savefig(png_path, dpi=400, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    
    print(f"\nSaved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")
    plt.close(fig)


def create_class_statistics_comparison(domain: str = "xrf55", output_dir: Path = None):
    """Create a comprehensive statistics table comparing representations by class."""
    output_dir = output_dir or Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_root = Path(__file__).resolve().parents[3] / "Transfer_Learning" / "Data"
    
    if domain.lower() == "xrf55":
        prefix = "xrf55"
    else:
        prefix = "wise4car"
    
    # Load data
    features_1d = np.load(data_root / "XRF55" / f"{prefix}_features.npy")
    labels_1d = np.load(data_root / "XRF55" / f"{prefix}_labels.npy")
    names_1d = np.load(data_root / "XRF55" / f"{prefix}_label_names.npy", allow_pickle=True)
    
    features_acf = np.load(data_root / "ACF" / f"{prefix}_acf_features.npy")
    features_stft = np.load(data_root / "STFT" / f"{prefix}_stft_features.npy")
    
    # Compute statistics per class
    stats_summary = []
    
    for class_idx in range(len(names_1d)):
        mask = labels_1d == class_idx
        n_samples = np.sum(mask)
        
        class_1d = features_1d[mask]
        class_acf = features_acf[mask]
        class_stft = features_stft[mask]
        
        # 1D statistics
        energy_1d = np.mean(class_1d ** 2, axis=(1, 2))  # [N]
        
        # ACF statistics (variance of correlation across lags shows structure)
        acf_var = np.mean(np.var(class_acf, axis=1), axis=1)  # Variance per sample [N] -> mean
        
        # STFT statistics
        stft_var = np.mean(np.var(class_stft, axis=(1, 2)), axis=0)  # Variance per sample [N] -> mean
        
        stats_summary.append({
            "class": str(names_1d[class_idx]),
            "n_samples": n_samples,
            "1d_mean_energy": np.mean(energy_1d),
            "1d_energy_std": np.std(energy_1d),
            "acf_avg_variance": np.mean(acf_var),
            "stft_avg_variance": np.mean(stft_var),
        })
    
    # Print summary
    print("\n" + "="*100)
    print("CLASS-LEVEL REPRESENTATION STATISTICS")
    print("="*100)
    print(f"{'Class':<15} {'Samples':<10} {'1D Energy':<12} {'Energy Var':<12} {'ACF Var':<12} {'STFT Var':<12}")
    print("-"*100)
    
    for stat in stats_summary:
        print(f"{stat['class']:<15} {stat['n_samples']:<10} "
              f"{stat['1d_mean_energy']:<12.4f} {stat['1d_energy_std']:<12.4f} "
              f"{stat['acf_avg_variance']:<12.4f} {stat['stft_avg_variance']:<12.4f}")
    
    print("="*100)
    print("KEY INSIGHTS:")
    print("- ACF captures temporal structure (high variance = periodic patterns)")
    print("- STFT captures spectro-temporal distribution")
    print("- 1D alone mixes both, harder to separate")
    print("="*100 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze why ACF > STFT > 1D")
    parser.add_argument("--domain", choices=["xrf55", "wise4car"], default="xrf55",
                        help="Data domain")
    parser.add_argument("--class", dest="class_name", type=str, default="Waving",
                        help="Action class to analyze (e.g., Waving, Turning, Sitting)")
    parser.add_argument("--n_samples", type=int, default=3,
                        help="Number of samples to visualize")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Output directory")
    parser.add_argument("--stats", action="store_true",
                        help="Also generate per-class statistics table")
    
    args = parser.parse_args()
    
    try:
        # Load and visualize
        samples = load_representations(domain=args.domain, class_name=args.class_name, 
                                      n_samples=args.n_samples)
        print(f"\nLoaded {len(samples)} samples of class '{args.class_name}' from {args.domain}")
        
        output_path = Path(args.output_dir)
        create_comparison_figure(samples, output_path)
        
        # Optional statistics
        if args.stats:
            create_class_statistics_comparison(domain=args.domain, output_dir=output_path)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
