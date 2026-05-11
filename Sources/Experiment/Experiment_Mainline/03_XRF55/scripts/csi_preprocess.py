import os
import glob
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import resample, detrend

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

@dataclass
class PreprocessConfig:
    target_timesteps: int = 256               # fixed T after resampling (保持与目标域对齐)
    target_duration_s: float = 4.6            # align with WiSe4Car target window
    pooled_subcarriers: int = 16              # band pooling bins (F)
    use_stats_pool: bool = False              # if True, use statistical pooling
    clip_value: Optional[float] = None        # clip after z-score; None disables
    fuse_mode: str = "mean"                  # mean or median across RX/links
    random_seed: int = 42

# 0-based label ids for unified 6 classes
LABEL_TO_ID: Dict[str, int] = {
    "Sitting": 0,
    "Reaching": 1,
    "Turning": 2,
    "Bending": 3,
    "Waving": 4,
    "Using Phone": 5,
}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

# Map原始XRF动作id -> 目标6类名称（用于源域列表）
ACT_TO_TARGET: Dict[int, str] = {
    3: "Using Phone",       # using a phone
    31: "Waving",           # waving
    38: "Turning",          # turning
    46: "Turning",          # shaking head -> Turning
    47: "Turning",          # nodding -> Turning
    5: "Bending",           # picking something
    6: "Reaching",          # putting something on the table
    18: "Reaching",         # handing something to someone
    36: "Sitting",          # sitting down
}

# ----------------------------------------------------------------------------
# Loading helpers (针对 XRF55 列表文件)
# ----------------------------------------------------------------------------

def parse_list_line(line: str) -> Tuple[str, str, str]:
    parts = line.strip().split(",")
    if len(parts) < 3:
        raise ValueError(f"Invalid line: {line!r}")
    filename, vol = parts[0], parts[1]
    # 支持两种格式：
    # 1) 源域：filename,vol,act_id
    # 2) 目标域（XRF_target）：filename,vol,label_name,label_id
    if len(parts) >= 4:
        label_name = parts[2].strip()
    else:
        act = int(parts[2])
        label_name = ACT_TO_TARGET.get(act)
        if label_name is None:
            raise ValueError(f"Unmapped act id: {act}")
    return filename, vol, label_name


def load_xrf_split(list_path: str, data_dir: str, act_to_target: Dict[int, str]) -> List[Tuple[np.ndarray, int]]:
    """
    兼容源域列表与 XRF_target 列表，返回 (csi_amplitude, label_id)
    data_dir: 对应 train_data/WiFi 或 test_data/WiFi
    """
    samples: List[Tuple[np.ndarray, int]] = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            filename, _vol, label_name = parse_list_line(line)
            label_id = LABEL_TO_ID.get(label_name)
            if label_id is None:
                # 若未匹配，尝试大小写归一
                label_id = LABEL_TO_ID.get(label_name.strip().title())
            if label_id is None:
                continue
            npy_path = os.path.join(data_dir, f"{filename}.npy")
            raw = np.load(npy_path)
            csi = np.abs(raw)
            samples.append((csi, label_id))
    return samples

# ----------------------------------------------------------------------------
# Core transforms
# ----------------------------------------------------------------------------

def fuse_channels(csi: np.ndarray, mode: str = "mean") -> np.ndarray:
    """
    Fuse multi-RX/multi-link to [time, sub].
    Acceptable input shapes (adapt reorder logic as needed):
      - [time, rx, sub]
      - [rx, time, sub] (will be moved to time-first)
      - [link, rx, time, sub] (links fused first)
    """
    arr = csi
    if arr.ndim == 2:
        return arr  # already [time, sub]
    if arr.ndim == 4:
        arr = arr.mean(axis=0)  # fuse links
    if arr.ndim == 3 and arr.shape[0] <= 8:  # heuristic: small first dim is rx
        arr = np.moveaxis(arr, 0, 1)  # [time, rx, sub]
    if arr.ndim != 3:
        raise ValueError(f"Unexpected CSI shape after adjustment: {arr.shape}")
    if mode == "mean":
        fused = arr.mean(axis=1)
    elif mode == "median":
        fused = np.median(arr, axis=1)
    else:
        raise ValueError("mode must be 'mean' or 'median'")
    return fused  # [time, sub]

def align_time_window(x: np.ndarray, target_duration_s: float, fs_hz: Optional[float]) -> np.ndarray:
    """
    If fs_hz known, center-trim/pad to target_duration_s; else pass-through.
    """
    if fs_hz is None:
        return x
    target_len = int(round(target_duration_s * fs_hz))
    if x.shape[0] > target_len:
        start = (x.shape[0] - target_len) // 2
        return x[start:start + target_len]
    if x.shape[0] < target_len:
        pad = target_len - x.shape[0]
        left = pad // 2
        right = pad - left
        return np.pad(x, ((left, right), (0, 0)), mode="edge")
    return x

def resample_time_axis(x: np.ndarray, target_len: int) -> np.ndarray:
    return resample(x, target_len, axis=0)

def pool_subcarriers(x: np.ndarray, pooled_bins: int) -> np.ndarray:
    """
    Band pooling into pooled_bins bins (default 16).
    Input: [time, sub]; Output: [time, pooled_bins]
    """
    time_len, sub = x.shape
    if sub < pooled_bins:
        pad = pooled_bins - sub
        left = pad // 2
        right = pad - left
        x = np.pad(x, ((0, 0), (left, right)), mode="edge")
        sub = x.shape[1]
    bin_size = sub // pooled_bins
    pooled = []
    for b in range(pooled_bins):
        start = b * bin_size
        end = sub if b == pooled_bins - 1 else (b + 1) * bin_size
        pooled.append(x[:, start:end].mean(axis=1))
    return np.stack(pooled, axis=1)

def stats_pool_subcarriers(x: np.ndarray) -> np.ndarray:
    """Optional statistical features per time step."""
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    mn = x.min(axis=1)
    mx = x.max(axis=1)
    return np.stack([mean, std, mn, mx], axis=1)  # [time, 4]

def detrend_time(x: np.ndarray) -> np.ndarray:
    return detrend(x, axis=0, type="linear")

def normalize_sample(x: np.ndarray, clip_value: Optional[float]) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    sigma = x.std(axis=0, keepdims=True) + 1e-6
    z = (x - mu) / sigma
    if clip_value is not None:
        z = np.clip(z, -clip_value, clip_value)
    return z

# ----------------------------------------------------------------------------
# Dataset builder
# ----------------------------------------------------------------------------

def build_dataset(
    samples: List[Tuple[np.ndarray, int]],
    cfg: PreprocessConfig,
    domain_id: int,
    fs_hz: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    feats: List[np.ndarray] = []
    labels: List[int] = []
    domains: List[int] = []
    for csi, label_id in samples:
        x = fuse_channels(csi, mode=cfg.fuse_mode)          # [time, sub]
        x = align_time_window(x, cfg.target_duration_s, fs_hz)
        x = resample_time_axis(x, cfg.target_timesteps)     # [T, sub]
        if cfg.use_stats_pool:
            x = stats_pool_subcarriers(x)                   # [T, 4]
        else:
            x = pool_subcarriers(x, cfg.pooled_subcarriers) # [T, F]
        x = detrend_time(x)
        x = normalize_sample(x, clip_value=cfg.clip_value)
        feats.append(x.astype(np.float32))
        labels.append(label_id)
        domains.append(domain_id)
    return (
        np.stack(feats, axis=0),
        np.array(labels, dtype=np.int64),
        np.array(domains, dtype=np.int64),
    )

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    cfg = PreprocessConfig()
    np.random.seed(cfg.random_seed)

    # 处理 XRF_target（已映射到共享标签），若需源域可改回 XRF_dataset
    xrf_root = os.path.join("dataset", "XRF_target")
    train_list = os.path.join(xrf_root, "train_list.txt")
    val_list = os.path.join(xrf_root, "val_list.txt")
    train_dir = os.path.join(xrf_root, "train_data", "WiFi")
    val_dir = os.path.join(xrf_root, "test_data", "WiFi")

    source_fs = None  # 如果知道采样率，填入 Hz；否则保持 None

    train_samples = load_xrf_split(train_list, train_dir, ACT_TO_TARGET)
    val_samples = load_xrf_split(val_list, val_dir, ACT_TO_TARGET)

    train_X, train_y, train_d = build_dataset(train_samples, cfg, domain_id=0, fs_hz=source_fs)
    val_X, val_y, val_d = build_dataset(val_samples, cfg, domain_id=0, fs_hz=source_fs)

    X = np.concatenate([train_X, val_X], axis=0)
    y = np.concatenate([train_y, val_y], axis=0)
    d = np.concatenate([train_d, val_d], axis=0)
    label_names = np.array([ID_TO_LABEL[i] for i in range(len(ID_TO_LABEL))])

    uniq_labels = np.unique(y)
    if len(uniq_labels) != 6:
        raise ValueError(f"期望6类标签，实际找到 {len(uniq_labels)} 类: {uniq_labels}")

    out_dir = os.path.join("preprocessed")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "XRF55_features.npy"), X)
    np.save(os.path.join(out_dir, "XRF55_labels.npy"), y)
    np.save(os.path.join(out_dir, "XRF55_domains.npy"), d)
    np.save(os.path.join(out_dir, "XRF55_label_names.npy"), label_names)
    # 方便区分 train/val，可额外保存拆分索引
    np.save(os.path.join(out_dir, "XRF55_train_indices.npy"), np.arange(len(train_y), dtype=np.int64))
    np.save(os.path.join(out_dir, "XRF55_val_indices.npy"), np.arange(len(train_y), len(train_y) + len(val_y), dtype=np.int64))

    print(
        f"Train kept: {len(train_y)}, Val kept: {len(val_y)}, Total: {X.shape[0]}, "
        f"shape={X.shape}, labels={uniq_labels.tolist()} -> {out_dir}"
    )

if __name__ == "__main__":
    main()
