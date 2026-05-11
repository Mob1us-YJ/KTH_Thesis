from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
from scipy import signal


EPS = 1e-6


@dataclass
class CommonConfig:
    target_timesteps: int = 256
    pooled_bands: int = 16
    detrend_mode: str = "demean"  # demean | linear
    clip_value: float | None = 6.0


@dataclass
class STFTConfig:
    nperseg: int = 64
    noverlap: int = 48
    nfft: int = 128
    use_power: bool = True
    use_log1p: bool = True
    clip_value: float | None = 6.0


@dataclass
class ACFConfig:
    max_lag: int = 64
    normalize_by_lag0: bool = True
    standardize_per_feature: bool = True
    clip_value: float | None = 6.0


def _resample_time_axis(x: np.ndarray, target_len: int) -> np.ndarray:
    if x.shape[0] == target_len:
        return x
    old_t = np.linspace(0.0, 1.0, num=x.shape[0], endpoint=True)
    new_t = np.linspace(0.0, 1.0, num=target_len, endpoint=True)
    return np.stack([np.interp(new_t, old_t, x[:, i]) for i in range(x.shape[1])], axis=1)


def _pool_subcarriers(x: np.ndarray, pooled_bands: int) -> np.ndarray:
    t_len, f_raw = x.shape
    if f_raw < pooled_bands:
        pad = pooled_bands - f_raw
        left = pad // 2
        right = pad - left
        x = np.pad(x, ((0, 0), (left, right)), mode="edge")
        f_raw = x.shape[1]

    edges = np.linspace(0, f_raw, pooled_bands + 1, dtype=int)
    pooled = []
    for i in range(pooled_bands):
        band = x[:, edges[i] : edges[i + 1]]
        pooled.append(band.mean(axis=1) if band.shape[1] else np.zeros((t_len,), dtype=x.dtype))
    return np.stack(pooled, axis=1)


def _detrend_or_demean(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "linear":
        return signal.detrend(x, axis=0, type="linear")
    return x - x.mean(axis=0, keepdims=True)


def _zscore_sample(x: np.ndarray, clip_value: float | None = None) -> np.ndarray:
    mu = x.mean()
    sigma = x.std() + EPS
    y = (x - mu) / sigma
    if clip_value is not None:
        y = np.clip(y, -clip_value, clip_value)
    return y


def preprocess_window_1d_common(x_raw: np.ndarray, cfg: CommonConfig) -> np.ndarray:
    """Common front-end used by both STFT and ACF.

    Input: x_raw [T_raw, F_raw]
    Output: x_1d [T_fixed, F_fixed] == [256, 16] by default
    """
    x = _resample_time_axis(x_raw, target_len=cfg.target_timesteps)
    x = _pool_subcarriers(x, pooled_bands=cfg.pooled_bands)
    x = _detrend_or_demean(x, mode=cfg.detrend_mode)
    x = _zscore_sample(x, clip_value=cfg.clip_value)
    return x.astype(np.float32)


def preprocess_window_for_stft(x_1d: np.ndarray, cfg: STFTConfig) -> np.ndarray:
    """Per-band STFT to keep spatial (band) + temporal-frequency information.

    Input: x_1d [T, F]
    Output: spec [C, H, W] where
      C = F (bands), H = nfft//2 + 1, W = num time frames
    """
    t_len, n_bands = x_1d.shape
    if cfg.noverlap >= cfg.nperseg:
        raise ValueError("STFT noverlap must be smaller than nperseg")
    if cfg.nperseg > t_len:
        raise ValueError("STFT nperseg cannot exceed window length")

    per_band = []
    for b in range(n_bands):
        _, _, z = signal.stft(
            x_1d[:, b],
            window="hann",
            nperseg=cfg.nperseg,
            noverlap=cfg.noverlap,
            nfft=cfg.nfft,
            boundary=None,
            padded=False,
            return_onesided=True,
        )
        s = np.abs(z)
        if cfg.use_power:
            s = s ** 2
        if cfg.use_log1p:
            s = np.log1p(s)
        per_band.append(s.astype(np.float32))

    spec = np.stack(per_band, axis=0)
    spec = _zscore_sample(spec, clip_value=cfg.clip_value)
    return spec.astype(np.float32)


def build_stft_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    domain_id: int,
    common_cfg: CommonConfig,
    stft_cfg: STFTConfig,
    assume_preprocessed_1d: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build STFT tensor set from window features.

    If assume_preprocessed_1d=True, features are assumed [N, 256, 16] and passed
    directly to STFT. Otherwise each sample is first passed through
    preprocess_window_1d_common.
    """
    out = []
    for i in range(features.shape[0]):
        x = features[i]
        x_1d = x if assume_preprocessed_1d else preprocess_window_1d_common(x, common_cfg)
        out.append(preprocess_window_for_stft(x_1d, stft_cfg))

    x_stft = np.stack(out, axis=0).astype(np.float32)
    y = labels.astype(np.int64)
    d = np.full((len(y),), int(domain_id), dtype=np.int64)
    return x_stft, y, d


def compute_acf(x: np.ndarray, max_lag: int, normalize_by_lag0: bool = True) -> np.ndarray:
    """Normalized ACF on positive lags only.

    Input: x [T]
    Output: acf [max_lag + 1] for lags 0..max_lag
    """
    x0 = x - x.mean()
    corr = np.correlate(x0, x0, mode="full")
    mid = len(corr) // 2
    acf = corr[mid : mid + max_lag + 1].astype(np.float32)
    if normalize_by_lag0:
        acf = acf / (float(acf[0]) + EPS)
    return acf


def preprocess_window_for_acf(x_1d: np.ndarray, cfg: ACFConfig) -> np.ndarray:
    """Per-band ACF to preserve temporal + band structure.

    Input: x_1d [T, F]
    Output: acf_map [L, F], where L=max_lag+1
    """
    t_len, n_bands = x_1d.shape
    max_lag = min(cfg.max_lag, t_len - 1)
    acf_bands = [compute_acf(x_1d[:, b], max_lag=max_lag, normalize_by_lag0=cfg.normalize_by_lag0) for b in range(n_bands)]
    acf_map = np.stack(acf_bands, axis=1)

    if cfg.standardize_per_feature:
        mu = acf_map.mean(axis=0, keepdims=True)
        sigma = acf_map.std(axis=0, keepdims=True) + EPS
        acf_map = (acf_map - mu) / sigma

    if cfg.clip_value is not None:
        acf_map = np.clip(acf_map, -cfg.clip_value, cfg.clip_value)

    return acf_map.astype(np.float32)


def build_acf_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    domain_id: int,
    common_cfg: CommonConfig,
    acf_cfg: ACFConfig,
    assume_preprocessed_1d: bool = True,
    add_channel_dim: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    out = []
    for i in range(features.shape[0]):
        x = features[i]
        x_1d = x if assume_preprocessed_1d else preprocess_window_1d_common(x, common_cfg)
        out.append(preprocess_window_for_acf(x_1d, acf_cfg))

    x_acf = np.stack(out, axis=0).astype(np.float32)
    if add_channel_dim:
        x_acf = np.expand_dims(x_acf, axis=1)
    y = labels.astype(np.int64)
    d = np.full((len(y),), int(domain_id), dtype=np.int64)
    return x_acf, y, d


def save_outputs(prefix: str, out_dir: Path, feats: np.ndarray, labels: np.ndarray, domains: np.ndarray, label_names: Sequence[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{prefix}_features.npy", feats)
    np.save(out_dir / f"{prefix}_labels.npy", labels)
    np.save(out_dir / f"{prefix}_domains.npy", domains)
    np.save(out_dir / f"{prefix}_label_names.npy", np.array(label_names))


def _load_transfer_learning_1d(data_root: Path, domain: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if domain == "xrf55":
        x = np.load(data_root / "XRF55" / "XRF55_features.npy")
        y = np.load(data_root / "XRF55" / "XRF55_labels.npy")
        names = np.load(data_root / "XRF55" / "XRF55_label_names.npy", allow_pickle=True)
        return x, y, names
    if domain == "wise4car":
        x = np.load(data_root / "Wise4Car" / "wise4car_unified_features.npy")
        y = np.load(data_root / "Wise4Car" / "wise4car_unified_labels.npy")
        names = np.load(data_root / "Wise4Car" / "wise4car_unified_label_names.npy", allow_pickle=True)
        return x, y, names
    raise ValueError(f"Unsupported domain: {domain}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build STFT/ACF features from unified 1D CSI windows")
    p.add_argument("--data_root", default="Transfer_Learning/Data", help="Contains XRF55/ and Wise4Car/")
    p.add_argument("--out_root", default="Transfer_Learning/Data", help="Output root")
    p.add_argument("--max_samples", type=int, default=0, help="For smoke test: keep first K samples per domain (0=all)")
    p.add_argument("--stft_nperseg", type=int, default=64)
    p.add_argument("--stft_noverlap", type=int, default=48)
    p.add_argument("--stft_nfft", type=int, default=128)
    p.add_argument("--acf_max_lag", type=int, default=64)
    p.add_argument("--clip_value", type=float, default=6.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    out_root = Path(args.out_root)

    common_cfg = CommonConfig(clip_value=args.clip_value)
    stft_cfg = STFTConfig(
        nperseg=args.stft_nperseg,
        noverlap=args.stft_noverlap,
        nfft=args.stft_nfft,
        clip_value=args.clip_value,
    )
    acf_cfg = ACFConfig(max_lag=args.acf_max_lag, clip_value=args.clip_value)

    domains = [("xrf55", 0), ("wise4car", 1)]

    for domain_name, domain_id in domains:
        x_1d, y, label_names = _load_transfer_learning_1d(data_root, domain_name)

        if args.max_samples > 0:
            keep = min(args.max_samples, len(y))
            x_1d = x_1d[:keep]
            y = y[:keep]

        x_stft, y_stft, d_stft = build_stft_dataset(
            x_1d,
            y,
            domain_id=domain_id,
            common_cfg=common_cfg,
            stft_cfg=stft_cfg,
            assume_preprocessed_1d=True,
        )
        save_outputs(f"{domain_name}_stft", out_root / "STFT", x_stft, y_stft, d_stft, label_names)

        x_acf, y_acf, d_acf = build_acf_dataset(
            x_1d,
            y,
            domain_id=domain_id,
            common_cfg=common_cfg,
            acf_cfg=acf_cfg,
            assume_preprocessed_1d=True,
            add_channel_dim=True,
        )
        save_outputs(f"{domain_name}_acf", out_root / "ACF", x_acf, y_acf, d_acf, label_names)

        print(
            f"{domain_name}: 1D={x_1d.shape}, STFT={x_stft.shape}, ACF={x_acf.shape}, labels={np.unique(y).tolist()}"
        )


if __name__ == "__main__":
    main()
