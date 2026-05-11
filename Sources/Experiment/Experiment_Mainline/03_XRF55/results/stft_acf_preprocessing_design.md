# Cross-Dataset CSI Preprocessing Design: STFT and ACF

## 1) Design Goal
Build STFT and ACF representations that are directly comparable between XRF55 (source) and WiSe4Car (target), while preserving the existing 1D pipeline philosophy:
- amplitude-only CSI
- same fixed window duration and temporal length
- same pooled band dimension
- same label space and ordering
- same normalization logic

## 2) Shared 1D Front-End (kept identical)
Input raw fused window: [T_raw, F_raw]
1. Resample time axis to T=256
2. Band pooling to F=16
3. Demean/detrend
4. Sample-wise z-score and optional clipping
Output unified 1D window: [256, 16]

In this project, STFT/ACF are generated from existing unified 1D arrays in Transfer_Learning/Data.

## 3) STFT Representation
Recommendation for transfer learning: compute STFT per pooled band (not on one aggregated 1D signal).
Reason: preserves both temporal-frequency cues and band-wise spatial structure, reducing information loss and improving cross-domain robustness.

Pipeline per window:
- For each band (16 total), apply STFT on 1D sequence length 256
- Window: Hann
- nperseg=64, noverlap=48 (hop=16), nfft=128
- Magnitude -> power (|Z|^2)
- log1p scaling
- Per-sample z-score + optional clipping

Output shape:
- per sample: [C, H, W] = [16, 65, 13]
- dataset: [N, 16, 65, 13]

## 4) ACF Representation
Recommendation for transfer learning: compute ACF per pooled band (not only aggregated signal).
Reason: keeps lag dynamics + band structure, better aligned with [T,F] representation than single-signal ACF.

Pipeline per window:
- For each band, compute normalized autocorrelation
- Keep positive lags only (including lag 0)
- max_lag=64 -> L=65
- Normalize by lag0
- Per-feature standardization across lags + optional clipping

Output shape:
- per sample: [L, F] = [65, 16]
- add channel dim for CNN compatibility: [1, 65, 16]
- dataset: [N, 1, 65, 16]

## 5) Cross-Dataset Alignment Rules
Keep these identical across XRF55 and WiSe4Car:
- window duration and temporal target length (T=256)
- pooled band count (F=16)
- STFT params (window/hop/nfft/log/power)
- ACF params (max_lag, normalization, clipping)
- label ordering: [Sitting, Reaching, Turning, Bending, Waving, Using Phone]
- domain_id convention: source=0 (XRF55), target=1 (WiSe4Car)

## 6) Generated Files
STFT:
- Transfer_Learning/Data/STFT/xrf55_stft_features.npy
- Transfer_Learning/Data/STFT/xrf55_stft_labels.npy
- Transfer_Learning/Data/STFT/xrf55_stft_domains.npy
- Transfer_Learning/Data/STFT/xrf55_stft_label_names.npy
- Transfer_Learning/Data/STFT/wise4car_stft_features.npy
- Transfer_Learning/Data/STFT/wise4car_stft_labels.npy
- Transfer_Learning/Data/STFT/wise4car_stft_domains.npy
- Transfer_Learning/Data/STFT/wise4car_stft_label_names.npy

ACF:
- Transfer_Learning/Data/ACF/xrf55_acf_features.npy
- Transfer_Learning/Data/ACF/xrf55_acf_labels.npy
- Transfer_Learning/Data/ACF/xrf55_acf_domains.npy
- Transfer_Learning/Data/ACF/xrf55_acf_label_names.npy
- Transfer_Learning/Data/ACF/wise4car_acf_features.npy
- Transfer_Learning/Data/ACF/wise4car_acf_labels.npy
- Transfer_Learning/Data/ACF/wise4car_acf_domains.npy
- Transfer_Learning/Data/ACF/wise4car_acf_label_names.npy

## 7) Implementation Entry
Script:
- Transfer_Learning/preprocess_stft_acf.py

Default full run:
- python Transfer_Learning/preprocess_stft_acf.py

Smoke test:
- python Transfer_Learning/preprocess_stft_acf.py --max_samples 8
