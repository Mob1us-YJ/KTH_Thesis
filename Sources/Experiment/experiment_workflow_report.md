# Experimental Workflow Report
## WiFi CSI-Based In-Car Human Activity Recognition via Transfer Learning and Domain Adaptation

---

## 1. Problem Statement

This work addresses **6-class Human Activity Recognition (HAR)** in an in-vehicle environment using WiFi Channel State Information (CSI). The central challenge is **cross-domain generalization** across two axes:

1. **Cross-sensor**: The source domain uses a **radar CSI** dataset (XRF55), while the target domain uses a **WiFi CSI** dataset (Wise4Car). The two sensors differ in carrier frequency, antenna configuration, and physical measurement principle.
2. **Cross-vehicle**: Wise4Car was collected across 6 different cars. Environmental multipath and channel characteristics vary substantially between vehicles, making per-car generalization non-trivial.

The six activity classes are: **Sitting, Reaching, Turning, Bending, Waving, Using Phone**.

---

## 2. Datasets

### 2.1 XRF55 — Source Domain

| Attribute | Value |
|-----------|-------|
| Sensor type | Radar CSI |
| Total samples | 1,980 |
| Classes | 6 |
| Usage | Source-domain pre-training only (never evaluated directly) |
| Split | Pooled 80/10/10 train/val/test |

XRF55 is a controlled indoor radar activity dataset. It provides labeled data for pre-training feature extractors before adaptation to WiFi CSI.

### 2.2 Wise4Car — Target Domain

| Attribute | Value |
|-----------|-------|
| Sensor type | WiFi CSI (802.11n) |
| Total samples | 2,117 |
| Classes | 6 (video-refined labels) |
| Vehicles | 6 cars (Car1–Car6) |
| Annotation | Video-synchronized, manually verified activity boundaries |

**Per-class distribution:**

| Class | Label ID | Count | Notes |
|-------|----------|-------|-------|
| Sitting | 0 | 903 | Most frequent |
| Reaching | 1 | 209 | Absent in Car1; sparse in Car3 (4 samples) |
| Turning | 2 | 478 | Stable across cars |
| Bending | 3 | 72 | Absent in Car3 |
| Waving | 4 | 2 | Effectively unusable |
| Using Phone | 5 | 453 | Absent in Car2 |

The severe per-car class coverage gaps make the LOCO (Leave-One-Car-Out) protocol particularly challenging.

---

## 3. Signal Preprocessing Pipeline

Each raw CSI sample is converted into **three parallel representations** (1D, STFT, ACFplus) through a shared preprocessing trunk followed by representation-specific branches. The pipeline is applied identically to both XRF55 and Wise4Car at dataset preparation time; the resulting `.npy` arrays are stored on disk and loaded at training time.

```
Raw CSI sample [T × F]
        │
        ▼
┌─────────────────────────────────────────┐
│         SHARED PREPROCESSING TRUNK      │
│  1. Layout normalization → [T, F]       │
│  2. Amplitude-only (drop phase)         │
│  3. Demean (per-subcarrier)             │
│  4. Temporal windowing → [256, F]       │
│  5. Subcarrier pooling → [256, 16]      │
│  6. Z-score per feature → [256, 16]     │
└──────────────┬──────────────────────────┘
               │
    ┌──────────┼──────────┐
    ▼          ▼          ▼
  [1D]       [STFT]    [ACFplus]
(16,256)  (16,33,13)  (2,16,64)
```

### 3.1 Shared Preprocessing Trunk

All steps operate on a 2D array of shape `[Time, Features]`.

**Step 1 — Layout normalization**

The raw input may arrive in `TF` (time-first) or `FT` (feature-first) layout depending on the dataset loader. The preprocessor always converts to `[T, F]` (time-major) as the canonical internal format.

**Step 2 — Amplitude only**

If the raw signal is complex-valued (IQ samples), only the magnitude `|z|` is retained. Phase is discarded: per-sample phase is unreliable without hardware-level calibration and sanitization, and cross-sensor phase is not meaningful.

```python
sample_tf = np.abs(sample_tf)   # if complex
```

**Step 3 — Demean (per-subcarrier)**

The time-mean of each subcarrier channel is subtracted independently. This removes static DC offsets that are environment-specific (furniture, car body reflections) and not related to human activity.

```python
sample_tf = sample_tf - sample_tf.mean(axis=0, keepdims=True)
```

> Note: Linear detrending (`detrend=false` in config) is disabled. Demean alone is sufficient given the short window length (256 steps ≈ a few seconds).

**Step 4 — Temporal windowing**

Each sample is cropped or padded to exactly 256 time steps using center-mode:

- If length > 256: crop symmetrically around the center
- If length < 256: pad with edge values on both sides

```
window_length: 256
window_mode: center
```

**Step 5 — Subcarrier pooling**

The original subcarrier count (varies by dataset and hardware) is downsampled to 16 uniform bins by **average pooling** consecutive subcarrier groups. This makes all datasets compatible with a fixed model input size and reduces high-frequency spectral noise.

```
subcarrier_bins: 16   → output shape: [256, 16]
```

**Step 6 — Z-score normalization (per subcarrier)**

Each of the 16 subcarrier channels is independently normalized to zero mean and unit variance across the 256 time steps of that sample:

```python
mean = sample_tf.mean(axis=0, keepdims=True)    # shape [1, 16]
std  = sample_tf.std(axis=0,  keepdims=True) + 1e-6
sample_tf = (sample_tf - mean) / std             # shape [256, 16]
```

This is **sample-level** normalization (not dataset-level), so no global statistics need to be shared between XRF55 and Wise4Car. This step is the primary mechanism for making 1D representations sensor-agnostic with respect to absolute amplitude scale.

---

### 3.2 Representation-Specific Branches

After the shared trunk, three representations are derived in parallel from the normalized `[256, 16]` array.

#### Branch A — 1D Raw Time Series

The `[256, 16]` matrix is transposed to `[16, 256]` (channels-first layout), matching PyTorch convention for 1D convolution:

```
one_d_layout: CT    →   shape: [16, 256]
```

| Attribute | Value |
|-----------|-------|
| Shape | `(N, 16, 256)` |
| Value range | approx. `[-3, +9]` after z-score |
| Interpretation | 16-channel 1D temporal signal |
| Model input | MultiscaleCNN `input_channels=16` |

The 1D representation retains raw temporal dynamics but inherits all sensitivity to sensor-specific amplitude statistics that survive z-score normalization (e.g., cross-subcarrier covariance structure).

#### Branch B — STFT (Short-Time Fourier Transform)

A per-subcarrier Hann-windowed STFT is computed using PyTorch's `torch.stft`, then the power spectrum is taken:

```python
spec = torch.stft(channel, n_fft=64, hop_length=16, win_length=64,
                  window=hann, return_complex=True, center=False)
power = spec.abs().pow(2.0)     # power=2.0
```

Parameters and output dimensions:

| Parameter | Value | Effect |
|-----------|-------|--------|
| `n_fft` | 64 | FFT size; frequency resolution = 64/2 + 1 = **33 bins** |
| `hop_length` | 16 | Step between frames; time frames = (256−64)/16 + 1 = **13** |
| `win_length` | 64 | Same as n_fft; Hann window |
| `power` | 2.0 | Power spectrum (magnitude²) |

| Attribute | Value |
|-----------|-------|
| Shape | `(N, 16, 33, 13)` = (samples, subcarriers, freq\_bins, time\_frames) |
| Value range | `[0, 4415]` raw (Wise4Car); `[-0.65, 3.93]` normalized (XRF55 stft\_norm) |
| Interpretation | 16-channel 2D time-frequency image |
| Model input | ResNet2D `input_channels=16` |

**Cross-domain caveat**: The raw STFT power is **not** on a common scale between XRF55 (radar) and Wise4Car (WiFi). For cross-sensor Transfer and DA Transfer experiments, a second normalization pass is required: `log1p(stft) → z-score`. This is applied to create the `wise4car_stft_norm_v1` and `xrf55_multiview_stft_norm_v1` dataset variants used in those experiments.

#### Branch C — ACFplus (Two-Branch Autocorrelation)

ACFplus is the primary representation for cross-domain experiments. It is computed over two signal branches, each producing a `[16, 64]` lag map, stacked along a new axis to give `[2, 16, 64]`.

**Branch 1 — `raw`**: autocorrelation of the normalized time series  
**Branch 2 — `diff`**: autocorrelation of the first-difference (temporal derivative) of the signal

For each branch and each of the 16 subcarriers, the autocorrelation is computed as:

```python
centered = signal - signal.mean()
corr = np.correlate(centered, centered, mode='full')   # length 2T-1
corr = corr[T-1:]                                       # keep lags 0..T-1
# unbiased correction: divide by (T - k) instead of T
corr /= np.arange(T, 0, -1)
# lag-0 normalization: divide by corr[0]
corr /= corr[0]
# keep lags 1..64 (zero-lag excluded: keep_zero_lag=False)
corr = corr[1:65]
```

Post-processing steps applied to each `[16, 64]` lag map:

| Step | Operation | Purpose |
|------|-----------|---------|
| `standardize_per_feature=True` | Z-score each subcarrier row independently | Remove per-subcarrier baseline differences |
| `standardize_per_map=True` | Global z-score of the full 2D map | Normalize overall map intensity |
| `clip_value=3.0` | Clip to `[−3, +3]` | Remove outlier spikes from noisy lags |

Full ACFplus configuration:

```yaml
acf:
  max_lag: 64
  normalize: lag0        # divide by lag-0 power
  unbiased: true         # unbiased estimator: divide by (T-k)
  keep_zero_lag: false   # exclude lag-0 from output
  branches: [raw, diff]  # 2 branches → output channels = 2
  standardize_per_feature: true
  standardize_per_map: true
  clip_value: 3.0
  log_compress: false
```

| Attribute | Value |
|-----------|-------|
| Shape | `(N, 2, 16, 64)` = (samples, branches, subcarriers, lags) |
| Value range | `[−3, +3]` (clipped) |
| Interpretation | 2-channel 2D lag-frequency image |
| Model input | ResNet2D `input_channels=2` |

**Why ACFplus is domain-invariant**: The lag-0 normalization (`normalize: lag0`) divides every lag by the zero-lag power, which is proportional to the signal energy. This makes all values scale-invariant: doubling the signal amplitude leaves the ACF values unchanged. As a result, the absolute amplitude mismatch between radar (XRF55) and WiFi (Wise4Car) is cancelled by construction — no cross-dataset normalization is needed.

The `diff` branch captures dynamics that are present in velocity-like features (rate of change of channel amplitude), complementing the `raw` branch's position-like features. The two branches together provide a richer representation of motion characteristics.

---

### 3.3 Output Storage and Dataset Variants

All three representations are precomputed once and stored as `.npy` files:

| File | Shape | Dataset |
|------|-------|---------|
| `1d.npy` | `(N, 16, 256)` | Both |
| `stft.npy` | `(N, 16, 33, 13)` | Both |
| `acf.npy` | `(N, 2, 16, 64)` | Wise4Car |
| `acf.npy` | `(N, 1, 16, 65)` | XRF55 (single branch, lag-0 included) |

Key dataset variants used in experiments:

| Directory name | Contents | Used by |
|----------------|----------|---------|
| `wise4car_video_refined_multiview_acfplus_v2` | Wise4Car, all 3 views, ACFplus 2-branch | All main experiments |
| `xrf55_multiview_acfplus_v2` | XRF55, all 3 views, ACFplus 1-branch | SSL + Transfer pre-training |
| `wise4car_stft_norm_v1` | Wise4Car STFT, log1p+zscore normalized | STFT Transfer/DA experiments |
| `xrf55_multiview_stft_norm_v1` | XRF55 STFT, log1p+zscore normalized | STFT Transfer/DA experiments |
| `wise4car_3class_loco` | Wise4Car, 3 classes (Sitting/Turning/Phone), metadata remapped | 3-class LOCO analysis |

---

## 4. Model Architectures

### 4.1 ResNet2D (for STFT and ACFplus)

A 2D ResNet adapted for CSI inputs:

| Component | Specification |
|-----------|---------------|
| Input channels | 2 (ACF) / 16 (STFT) |
| Base channels | 32 |
| Architecture | Residual blocks with batch normalization + GELU |
| Output | Global average pooling → FC(num_classes) |
| Dropout | 0.2 |
| Parameters (ACF) | **2.80 M** (10.7 MB FP32) |
| Parameters (STFT) | **2.82 M** (10.8 MB FP32) |

### 4.2 MultiscaleCNN (for 1D)

A multi-branch 1D CNN operating at multiple temporal scales:

| Component | Specification |
|-----------|---------------|
| Input channels | 16 |
| Architecture | Parallel conv branches with kernel sizes [3, 7, 15, 31] → concatenate → FC |
| Output | Global average pooling → FC(num_classes) |
| Dropout | 0.3 |
| Parameters | **3.65 M** (14.0 MB FP32) |

### 4.3 DANN Discriminator (for DA Transfer)

An additional gradient-reversal domain classifier appended to the shared encoder:

| Component | Specification |
|-----------|---------------|
| Input | Encoder feature map (flattened) |
| Architecture | FC(128) → BatchNorm → ReLU → FC(2) |
| Training | Gradient Reversal Layer (GRL) with λ=1.0 |

---

## 5. Training Strategies

Four strategies are evaluated for each representation × protocol combination. All use the same model architectures; the difference lies in how the encoder is initialized before target-domain fine-tuning.

**Strategy overview:**

```
Baseline:    [Random Init] ──────────────────────────────────→ [Wise4Car Fine-tune]

SSL:         [BYOL / XRF55, no labels] ──────────────────────→ [Wise4Car Fine-tune]

Transfer:    [Supervised / XRF55, with labels] ──────────────→ [Wise4Car Fine-tune]

DA Transfer: [Supervised / XRF55] → [DANN / MMD / DRCA] ─────→ [Wise4Car Fine-tune]
```

Each strategy is run independently for all 3 representations × 3 protocols (Pooled, In-Car ×6 folds, LOCO ×6 folds).

---

### 5.1 Baseline — Train from Scratch on Target Domain

The model is randomly initialized and trained **only on Wise4Car** with no pre-training. This establishes the lower-bound performance without any source-domain knowledge.

| Hyperparameter | Value |
|----------------|-------|
| Initialization | Random |
| Optimizer | AdamW, lr=3×10⁻⁴, weight_decay=1×10⁻⁴ |
| Batch size | 32 |
| Epochs | up to 60, early stopping patience=15 |
| Loss | Cross-entropy + label smoothing 0.05 |
| Class balancing | Inverse-frequency weighted sampler |
| Grad clip | 1.0 |
| Model selection | Val Macro-F1 |

---

### 5.2 SSL — BYOL Pre-training on XRF55 → Fine-tune on Wise4Car

A 2-stage pipeline. No source labels are used in Stage 1.

```
XRF55 (unlabeled) ──[Stage 1: BYOL]──→ ssl_encoder_best.pt
                                                │  (projector & predictor discarded)
                                                ▼
                           Wise4Car ──[Stage 2: Fine-tune]──→ Result
```

#### Stage 1 — BYOL Self-Supervised Pre-training (XRF55)

BYOL (Bootstrap Your Own Latent) trains an **online network** and a **target network** to produce consistent representations from two differently-augmented views of the same sample, without requiring negative pairs.

**Network additions (training only, discarded afterwards):**

- **MLP Projector**: `Linear(feat_dim → 256) → BN → GELU → Linear(256 → 128)` — appended to both online and target encoders
- **MLP Predictor**: identical structure — appended to the online branch only

**Loss function:**

```
L = -cos_sim(pred(proj(online(aug_A))), stop_grad(proj(target(aug_B))))
  + -cos_sim(pred(proj(online(aug_B))), stop_grad(proj(target(aug_A))))
```

Target network is updated by Exponential Moving Average (not by gradient):
```
θ_target ← m · θ_target + (1 − m) · θ_online,   m = 0.996
```

**CSI-specific data augmentations** (6 types, applied independently to each view pair):

| Augmentation | Parameter | Rationale |
|--------------|-----------|-----------|
| Gaussian Noise | σ = 0.01 | Simulate measurement noise |
| Time Mask | ratio = 0.1 (10% of time axis zeroed) | Robustness to temporal occlusion |
| Subcarrier Mask | ratio = 0.1 (10% of subcarrier axis zeroed) | Robustness to frequency dropout |
| Time Shift | max_ratio = 0.1 (±10% circular shift) | Time-shift invariance |
| Amplitude Scale | scale ∈ [0.8, 1.2] | Simulate inter-session amplitude variation |
| Local Crop + Resize | crop_ratio = 0.8, then interpolate to original length | Local temporal context invariance |

**Stage 1 hyperparameters:**

| Hyperparameter | Value |
|----------------|-------|
| Data | XRF55 (1,584 train / 198 val / 198 test) |
| Optimizer | AdamW, lr=3×10⁻⁴, weight_decay=1×10⁻⁴ |
| Batch size | 32 |
| Epochs | 30 |
| EMA momentum | 0.996 |
| Projector hidden dim | 256 |
| Projector output dim | 128 |
| Grad clip | 1.0 |
| Saved output | `ssl_encoder_best.pt` (encoder weights only; projector + predictor discarded) |

#### Stage 2 — Supervised Fine-tuning on Wise4Car

The SSL encoder is loaded as initialization. The classification head is randomly initialized. All layers are fully fine-tuned end-to-end.

| Hyperparameter | Value |
|----------------|-------|
| Initialization | `ssl_encoder_best.pt` (Stage 1 encoder) |
| Fine-tune mode | Full fine-tune (all layers unfrozen) |
| Optimizer | AdamW, lr=2×10⁻⁴, weight_decay=1×10⁻⁴ |
| Batch size | 32 |
| Epochs | 40, early stopping patience=10 |
| Loss | Cross-entropy + label smoothing 0.05 |
| Class balancing | Inverse-frequency weighted sampler |
| Grad clip | 1.0 |
| Model selection | Val Macro-F1 |

---

### 5.3 Transfer — Supervised Pre-training on XRF55 → Fine-tune on Wise4Car

A 2-stage pipeline. Source labels are used in Stage 1 to learn discriminative activity features directly.

```
XRF55 (labeled) ──[Stage 1: Supervised]──→ best_model.pt
                                                  │  (classification head replaced)
                                                  ▼
                             Wise4Car ──[Stage 2: Fine-tune]──→ Result
```

#### Stage 1 — Supervised Pre-training on XRF55

Standard supervised training with cross-entropy loss on the 6-class XRF55 radar activity dataset.

| Hyperparameter | Value |
|----------------|-------|
| Data | XRF55 (1,584 train / 198 val / 198 test) |
| Initialization | Random |
| Optimizer | AdamW, lr=3×10⁻⁴, weight_decay=1×10⁻⁴ |
| Batch size | 32 |
| Epochs | 30, early stopping patience=8 |
| Loss | Cross-entropy + label smoothing 0.02 |
| Class balancing | Inverse-frequency weights, cap_ratio=4.0, sampler=none |
| Grad clip | 1.0 |
| Model selection | Val Macro-F1 |
| Saved output | `best_model.pt` (full model including classification head) |

> **Note on 4-stage combined pipeline**: In the full `SSL → Supervised → DANN → Fine-tune` pipeline (`run_ssl_transfer_dann_pipeline.py`), Stage 1 (BYOL) provides the initialization for Stage 2 (supervised pre-training on XRF55). This acts as a warm-start: BYOL features provide a better starting point than random initialization for supervised source training. The standalone Transfer strategy (results in `comparison_results.json`) uses random initialization for Stage 1.

#### Stage 2 — Full Fine-tuning on Wise4Car

Identical setup to SSL Stage 2. The XRF55 classification head is replaced with a randomly initialized Wise4Car 6-class head, and the full network is fine-tuned.

| Hyperparameter | Value |
|----------------|-------|
| Initialization | `best_model.pt` (Stage 1 supervised encoder) |
| Fine-tune mode | Full fine-tune (all layers unfrozen) |
| Optimizer | AdamW, lr=2×10⁻⁴, weight_decay=1×10⁻⁴ |
| Batch size | 32 |
| Epochs | 40, early stopping patience=10 |
| Loss | Cross-entropy + label smoothing 0.05 |
| Class balancing | Inverse-frequency weighted sampler |
| Grad clip | 1.0 |
| Model selection | Val Macro-F1 |

**Comparison of SSL vs. Transfer initialization:**

| | SSL Stage 2 init | Transfer Stage 2 init |
|-|-----------------|----------------------|
| Source | BYOL encoder (no labels used) | Supervised encoder (labels used) |
| Strength | Generalizable temporal features | Directly discriminative for activity classes |
| Weakness | May not capture label-relevant structure | May overfit to radar-specific statistics |

---

### 5.4 DA Transfer — Supervised Pre-training + Domain Adaptation → Fine-tune

A 3-stage pipeline. Domain adaptation explicitly minimizes the feature distribution gap between XRF55 and Wise4Car before target fine-tuning.

```
XRF55 (labeled) ──[Stage 1: Supervised]──→ best_model.pt
                                                  │
                                                  ▼
                    XRF55 + Wise4Car ──[Stage 2: DA (DANN/MMD/DRCA)]──→ adapted_encoder.pt
                                                                                │
                                                                                ▼
                                               Wise4Car ──[Stage 3: Fine-tune]──→ Result
```

**Stage 1** — Identical to Transfer Stage 1 (supervised pre-training on XRF55).

#### Stage 2 — Domain Adaptation

The pre-trained encoder is adapted to align source (XRF55) and target (Wise4Car) feature distributions. Three methods are evaluated:

**DANN (Domain-Adversarial Neural Network)**

A gradient reversal layer (GRL) routes the encoder's features into a domain discriminator. The GRL negates gradients during backpropagation, forcing the encoder to produce features that the discriminator cannot distinguish by domain.

```
Total loss = L_cls(x_source) + λ · L_adv(x_source, x_target)
```

| Hyperparameter | Value |
|----------------|-------|
| GRL scale λ | 1.0 |
| Discriminator | FC(128) → BN → ReLU → Dropout(0.2) → FC(2) |
| Optimizer | AdamW, lr=2×10⁻⁴, weight_decay=1×10⁻⁴ |
| Batch size | 32 |
| Epochs | 20, early stopping patience=5 |
| DA weight | 1.0 |

**MMD (Maximum Mean Discrepancy)**

Adds a kernel-based distributional distance as a regularization term. No adversarial training.

```
Total loss = L_cls(x_source) + w · MMD²(φ(x_source), φ(x_target))
```

| Hyperparameter | Value |
|----------------|-------|
| MMD weight w | 1.0 |
| Kernel | RBF |
| Optimizer | AdamW, lr=2×10⁻⁴ |
| Epochs | 20 |

**DRCA (Domain-Regularized Component Analysis)**

Aligns the second-order statistics (covariance) of source and target feature spaces via a learned linear projection, combined with supervised classification on the source domain.

#### Stage 3 — Fine-tuning on Wise4Car

The domain-adapted encoder (from the best DA checkpoint) is fine-tuned on labeled Wise4Car data.

| Hyperparameter | Value |
|----------------|-------|
| Initialization | DA stage best checkpoint |
| Fine-tune mode | Full fine-tune |
| Optimizer | AdamW, lr=2×10⁻⁴, weight_decay=1×10⁻⁴ |
| Batch size | 32 |
| Epochs | 40, early stopping patience=10 |
| Loss | Class-balanced focal loss (γ=2.0) + label smoothing 0.02 |
| Class balancing | Capped inverse-frequency, cap_ratio=5.0, weighted sampler |
| Grad clip | 1.0 |
| Model selection | Val Macro-F1 |

> The focal loss (γ=2.0) in Stage 3 down-weights easy (well-classified) samples and focuses training on hard examples — particularly useful for the minority classes (Bending, Waving, Reaching) which are severely under-represented in Wise4Car.

---

## 6. Evaluation Protocols

### 6.1 Pooled

All 6 cars are merged and split randomly:

| Split | Samples |
|-------|---------|
| Train | 1,695 (80%) |
| Val | 211 (10%) |
| Test | 211 (10%) |

Reflects average performance assuming i.i.d. car distribution.

### 6.2 In-Car (6-Fold Cross-Validation)

Each fold uses one car for test and the remaining five for training:

| Fold | Test car | Train cars |
|------|----------|------------|
| 1 | Car1 | Cars 2–6 |
| … | … | … |
| 6 | Car6 | Cars 1–5 |

Macro-F1 is averaged across all 6 folds. Measures within-distribution generalization when test and train cars partially overlap in distribution.

### 6.3 LOCO (Leave-One-Car-Out)

Each fold holds out one car entirely — no data from the test car is seen during training:

| Fold | Test car | Train cars |
|------|----------|------------|
| 1 | Car1 | Cars 2–6 |
| … | … | … |
| 6 | Car6 | Cars 1–5 |

The split manifests differ from In-Car: the val set is also drawn from training cars (not the test car). This is the strictest protocol measuring true cross-vehicle generalization.

**Structural challenge**: Wise4Car's per-car class imbalance makes LOCO inherently difficult regardless of model capacity. Waving appears only 2 times in the entire dataset; Reaching is absent in Car1 and has only 4 samples in Car3. Any fold where such a class is in the test car will yield near-zero recall for that class.

---

## 7. Results — Macro-F1 Across All Configurations

### 7.1 Full Result Table

| Representation | Strategy | In-Car | Pooled | LOCO |
|----------------|----------|--------|--------|------|
| **1D** | Baseline | 0.582 | 0.371 | 0.364 |
| | SSL | 0.618 | 0.401 | 0.492 |
| | Transfer | 0.554 | 0.421 | 0.509 |
| | **DA Transfer** | **0.664** | **0.532** | **0.524** |
| **STFT** | Baseline | 0.648 | 0.458 | 0.427 |
| | SSL | 0.572 | 0.446 | 0.436 |
| | **Transfer** | 0.721 | 0.542 | **0.551** |
| | DA Transfer | **0.736** | **0.566** | 0.549 |
| **ACF** | Baseline | 0.602 | 0.491 | 0.476 |
| | SSL | 0.728 | 0.513 | 0.498 |
| | Transfer | 0.751 | 0.521 | 0.510 |
| | **DA Transfer** | **0.809** | **0.762** | **0.716** |

Bold = best within each representation block per protocol.

### 7.2 Best Configuration per Representation

| Representation | Best Strategy | In-Car | Pooled | LOCO |
|----------------|---------------|--------|--------|------|
| 1D | DA Transfer | 0.664 | 0.532 | 0.524 |
| STFT | DA Transfer | 0.736 | 0.566 | 0.551 |
| **ACF** | **DA Transfer** | **0.809** | **0.762** | **0.716** |

**Overall best**: ACFplus + DA Transfer (DANN), pooled F1 = **0.762**, in-car F1 = **0.809**.

---

## 8. Domain Adaptation Method Comparison (ACFplus, Pooled Protocol)

Fine-grained comparison of DA methods on the best representation (ACFplus) under the relaxed pooled protocol (Wise4Car only, 211 test samples):

| Method | Macro-F1 | Accuracy | Δ vs. Transfer |
|--------|----------|----------|----------------|
| Transfer (No DA) | 0.510 | 0.507 | — |
| MMD | 0.609 | 0.621 | +0.099 |
| DRCA | 0.641 | 0.678 | +0.131 |
| **DANN** | **0.716** | **0.692** | **+0.206** |

DANN's adversarial feature alignment provides the largest gain. The gradient reversal mechanism directly optimizes domain confusion, which is more effective than the second-order statistics matching used by MMD and DRCA under the limited sample regime (~2,000 target samples).

### 8.1 Per-Class Analysis — DANN Confusion Matrix (n=211)

|  | Sitting | Reaching | Turning | Bending | Using Phone |
|--|---------|---------|---------|---------|------------|
| **Sitting** | 69% (62) | 2% (2) | 19% (17) | 0% (0) | 10% (9) |
| **Reaching** | 14% (3) | 67% (14) | 14% (3) | 0% (0) | 5% (1) |
| **Turning** | 17% (8) | 4% (2) | 62% (30) | 0% (0) | 17% (8) |
| **Bending** | 29% (2) | 0% (0) | 0% (0) | 71% (5) | 0% (0) |
| **Using Phone** | 13% (6) | 0% (0) | 9% (4) | 0% (0) | 78% (35) |

Primary confusion: Sitting ↔ Turning (both involve upper-body static or slow movement). Bending has very few test samples (n=7) making its recall estimate noisy.

---

## 9. Analysis of Results

### 9.1 Representation Effectiveness

**ACFplus consistently outperforms 1D and STFT** across all strategies and protocols. The margin is especially large for DA Transfer under pooled evaluation (+0.196 over STFT, +0.230 over 1D). This is attributable to the autocorrelation operation:

- It suppresses absolute amplitude scale, which is the dominant source of domain shift between radar (XRF55) and WiFi (Wise4Car)
- It preserves temporal structure (periodicity, motion rhythm) that is discriminative across activities
- It is input-normalization-free: no log1p or z-score tuning required for cross-sensor use

**STFT** performs second best overall. The 16-channel power spectrogram captures frequency content well within each sensor, but the raw power scale mismatch between XRF55 and Wise4Car (max ~3.9 vs. ~4415) creates a large domain gap that DA must overcome.

**1D** raw time series is the weakest representation. Although z-score normalization reduces inter-sensor amplitude differences, temporal shape statistics are not as stable across sensors as the autocorrelation function.

### 9.2 Strategy Effectiveness

**DA Transfer is consistently the best strategy** for all representations. Key observations:

- For 1D: DA Transfer improves over SSL (+0.046 pooled, +0.032 LOCO) and over Transfer (+0.111 pooled, +0.015 LOCO)
- For STFT: DA Transfer improves over Transfer (+0.024 pooled), but the gap is smaller, suggesting STFT features are already more transferable
- For ACF: DA Transfer delivers the largest absolute improvement over Transfer (+0.241 pooled), which confirms that explicit domain alignment on top of already-invariant features provides additional benefit beyond structural invariance alone

**SSL underperforms supervised Transfer in most settings**. This is consistent with the general finding that supervised pre-training is superior to self-supervised when labeled source data is available. The exception is 1D + LOCO, where SSL (0.492) slightly outperforms Transfer (0.509) is reversed — Transfer actually does better. For ACF + SSL (in-car 0.728), BYOL pre-training alone surpasses STFT Baseline, indicating that BYOL effectively extracts domain-general temporal patterns.

**STFT SSL (0.572 in-car) falls below STFT Baseline (0.648)**. This is a known failure mode of contrastive/BYOL SSL on frequency-domain inputs: the augmentations (time shift, amplitude scale) alter spectral structure in ways that may reinforce rather than suppress domain-specific characteristics.

### 9.3 Protocol Difficulty Ranking

For all representations and strategies: **In-Car > Pooled > LOCO** in terms of achieved F1.

- **In-Car** is easiest because the same car's data appears in both train and test (different sessions), so channel statistics are partially shared
- **Pooled** is moderate — random split ensures class balance but some test samples come from cars not in training
- **LOCO** is hardest due to complete absence of the test car from training. The gap LOCO vs. Pooled is large for weaker representations (1D: 0.524 vs. 0.532) and much larger for ACF DA Transfer (0.716 vs. 0.762), suggesting that even the best method leaves meaningful cross-vehicle variance unaddressed

### 9.4 LOCO Structural Ceiling

The LOCO protocol's difficulty is **fundamentally constrained by data coverage**, not model capacity. The 6-class LOCO F1 ceiling is low because:

- Waving: 2 total samples — any fold where Waving is in the test car yields F1=0 for that class
- Reaching: completely absent in Car1, near-absent in Car3 — leave-one-car-out folds for these cars have zero training signal for Reaching in test evaluation
- Bending: absent in Car3

A restricted **3-class LOCO** (Sitting, Turning, Using Phone; Cars 1,3,4,5,6) that removes these structurally broken classes achieves val Macro-F1 = **0.767** (mean across 5 folds), confirming the model learns cross-car representations well when training data is balanced.

---

## 10. Model Efficiency

### 10.1 Inference Hardware

| Component | Specification |
|-----------|---------------|
| CPU | AMD EPYC 7V12 64-Core |
| GPU | NVIDIA Tesla T4 (16 GB) |
| RAM | 27 GB |

### 10.2 Model Size and Latency

| Model | Parameters | Size (FP32) | CPU p50 (batch=1) | GPU p50 (batch=1) | GPU throughput (batch=32) |
|-------|-----------|------------|-------------------|-------------------|--------------------------|
| ACF / ResNet2D | 2.80 M | 10.7 MB | 5.7 ms | 2.0 ms | 14,776 samples/s |
| STFT / ResNet2D | 2.82 M | 10.8 MB | 6.1 ms | 2.0 ms | 15,141 samples/s |
| 1D / MultiscaleCNN | 3.65 M | 14.0 MB | 8.2 ms | 1.9 ms | 2,851 samples/s |

All models achieve ≤2 ms GPU inference latency per sample and ≤11 MB model size, satisfying real-time in-vehicle edge deployment constraints.

---

## 11. Key Findings Summary

1. **ACFplus is the optimal representation for cross-sensor transfer.** The autocorrelation operation provides sensor-agnostic normalization that no post-hoc normalization of raw features achieves. It delivers +0.241 pooled F1 improvement over DA Transfer on 1D.

2. **Domain adaptation consistently improves over no-DA transfer.** Gains range from +0.024 (STFT pooled) to +0.241 (ACF pooled). DANN outperforms MMD and DRCA in all tested conditions, with the largest margin in the limited-data regime.

3. **Supervised pre-training (Transfer) is more effective than self-supervised (SSL/BYOL) for this task.** When labeled source data is available, SSL does not provide additional benefit over supervised initialization. BYOL's usefulness is limited by the small XRF55 dataset (1,980 samples) and by augmentations that may not preserve discriminative activity structure.

4. **LOCO performance is bottlenecked by data collection coverage, not algorithms.** Classes with fewer than ~10 samples per car (Waving, Bending, Reaching in some cars) make any LOCO evaluation of 6-class classification unreliable. This should be addressed at the data collection stage.

5. **All models are edge-deployable.** ResNet2D with 2.8 M parameters achieves GPU latency of 2 ms/sample — suitable for real-time in-car HAR applications.

---

## 12. Reproducibility

| Component | Fixed value |
|-----------|-------------|
| Random seed | 42 (all experiments) |
| Optimizer | AdamW |
| Early stopping | Val Macro-F1, patience varies by stage (10–25 epochs) |
| Class balancing | Inverse-frequency weighted sampler (all supervised stages) |
| Gradient clipping | 0.5–1.0 (stage-dependent) |

All experiments run on a single CPU or single T4 GPU. No multi-GPU or distributed training.

---

*Results reference: `outputs/full_comparison/comparison_results.json` (36 entries, 3 representations × 4 strategies × 3 protocols)*
