# CSI 数据表示形式对比与应用

## 执行摘要

本模块通过可视化演示 WiFi CSI 信号在三个不同表示空间中的特性：

| 表示 | 形状 | 维度含义 | 适用场景 | 优势 |
|------|------|---------|---------|------|
| **1D Sequence** | [256, 16] | T=时间步, F=频带 | 基线模型 | 直观、信息完整、易于复原 |
| **STFT** | [16, 65, 13] | F=频带, H=频率,W=时间帧 | CNN/RNN | 时频分离、特征可视化 |
| **ACF** | [65, 16] | L=相关延迟, F=频带 | 周期性分析 | 捕捉运动周期、噪声鲁棒 |

## 1. 预处理统一前端 (Unified 1D Front-End)

所有三种表示都基于相同的统一 1D 预处理阶段：

```
原始窗口 [T_raw, F_raw]
    ↓
1. 时间轴重采样 → T=256
    ↓
2. 频带池化 → F=16 (16个均匀分布的子载波组)
    ↓
3. 去趋势/去均值 (demean or linear detrend)
    ↓
4. 逐样本 z-score 标准化 + ±6σ裁剪
    ↓
统一 1D 表示 [256, 16]
```

### 参数说明
- **目标时间长度 T=256**: 对应约 4.6 秒的 CSI 捕获（假设采样率 ~56 Hz）
- **目标频带数 F=16**: 折衷考虑计算复杂度 vs 频域分辨率
- **标准化策略**: 每个样本独立标准化，保留相对能量信息

## 2. 1D CSI 序列 (Time-Spectral)

### 定义
直接的 [256, 16] 时频矩阵：行为时间步，列为频带。

### 生成流程
```python
x_1d = preprocess_window_1d_common(x_raw, config)  # 输出 [256, 16]
```

### 特性
- **信息完整性**: 最小化信息损失（无谱分解）
- **可视化**: 易于理解为热力图（时间 vs 频带）
- **模型兼容性**: 适合 MLP、简单 LSTM
- **计算效率**: 最轻量级

### 应用案例
- ✅ 基准线（baseline）比对
- ✅ 特征池化或聚合（mean/std）
- ✅ 权重初始化
- ✅ 数据增强（mixup）

---

## 3. STFT 表示 (Spectro-Temporal)

### 定义
对每个频带独立应用 **短时傅里叶变换** (Short-Time Fourier Transform)。

输出: [F=16, H=65, W=13]
- F: 频带数（保留 1D 空间维）
- H: 频率箱数 (nfft//2+1 = 128//2+1 = 65)
- W: 时间帧数（由 hop_length 决定）

### 生成流程
```python
for each band b in [0, 16):
    f, t, Z = scipy.signal.stft(
        x_1d[:, b],
        window='hann',
        nperseg=64,
        noverlap=48,        # hop_length = 16
        nfft=128,
        return_onesided=True
    )
    
    # 后处理
    magnitude = |Z|        # 幅度谱
    power = magnitude^2    # 功率谱 (use_power=True)
    log_power = log1p(power)  # 对数缩放
    z_score_normalized_per_sample
    clip to ±6σ
```

### 参数解释
| 参数 | 值 | 含义 |
|------|------|------|
| `nperseg` | 64 | 分析窗口长度（样本数） |
| `noverlap` | 48 | 相邻窗口重叠样本数 |
| `hop_length` | 64 - 48 = 16 | 窗口步长 |
| `nfft` | 128 | FFT 填零到 128 点 |
| 时间帧数 W | (256 - 64) / 16 + 1 = 13 | 输出帧数计算 |

### 时间分辨率
- 原始 T=256 对应 ~4.6 秒
- STFT 时间帧数 W=13，相邻帧间隔 = 16 samples ≈ 286 ms
- 每帧覆盖 64 samples ≈ 1.14 秒

### 特性
- **时频分离**: 捕捉频谱随时间演化的动态
- **特征提取友好**: 易识别频率成分、谐波结构
- **噪声鲁棒性**: 平滑化作用（窗口重叠）
- **计算复杂度**: 中等（每个频带一个 FFT）

### 应用案例
- ✅ CNN 模型（2D 卷积）
- ✅ 时频注意力机制
- ✅ 多尺度融合（融合多个 STFT 参数配置）
- ✅ 谱特征工程（峰值检测、能量浓度）

### 可视化理解
```
原始 1D 信号         STFT 频谱图
(256 时间点)    →   (13 帧 × 65 频率点)

时间轴 →           时间帧 →
↓                 ↓
频带 1 ┌─────┐    频率 ┌──────────┐
频带 2 │ 1D  │    箱   │ 热力图   │
频带 3 │ 图  │    ↓    │ (热=高能)│
...    └─────┘         └──────────┘
```

---

## 4. ACF 表示 (Autocorrelation Function)

### 定义
对每个频带计算 **自相关函数** (Autocorrelation Function)，保留 65 个延迟值 (lags 0~64)。

输出: [L=65, F=16]
- L: 相关延迟数（从 lag=0 到 lag=64）
- F: 频带数

### 生成流程
```python
for each band b in [0, 16):
    x0 = x_1d[:, b]  # 该频带时间序列
    x0 = x0 - x0.mean()  # 去均值
    
    # 标准自相关计算
    acf = correlate(x0, x0, mode='full')
    acf = acf[mid:mid+65]  # 提取正延迟部分
    
    # 归一化（可选）
    if normalize_by_lag0:
        acf = acf / acf[0]  # 相对于 lag-0 (方差)
    
    # 标准化
    acf = (acf - mean) / (std + eps)
    clip to ±6σ
```

### 关键参数
| 参数 | 值 | 含义 |
|------|------|------|
| `max_lag` | 64 | 最大延迟（样本数）≈ 1.14 秒 |
| `normalize_by_lag0` | True | 除以方差（lag=0 值），归一化到 [-1, 1] |
| `standardize_per_feature` | True | 在 lag 维度上再进行 z-score |

### 时间尺度映射
- lag=0 → 同时相关（方差）
- lag=1 → 相邻样本相关（~17.9 ms 时延）
- lag=4 → ~71 ms 时延（可检测走路频率）
- lag=64 → ~1.14 秒时延

### 特性
- **周期性捕捉**: ACF 峰值对应信号周期
- **运动特征敏感**: 走路/转身/挥手等有不同的周期谱
- **噪声鲁棒**: 自相关对广义噪声鲁棒
- **可解释性**: 直观反映时间依赖性
- **计算高效**: O(T log T)，使用 FFT 加速

### 应用案例
- ✅ 周期性运动识别（走路、转身）
- ✅ 运动速度估计（峰值延迟位置）
- ✅ 序列模型初始化（特征工程）
- ✅ 领域泛化（周期特征跨数据集稳定）

### 可视化理解
```
原始 1D 序列          ACF 相关图
(256 时间点)  →    (65 个延迟 × 16 频带)

时间轴 →            延迟 (lag) →
↓                 ↓
频带 1 ┌─────┐    频带  ┌──────────┐
频带 2 │ 1D  │    16   │  热力图  │
频带 3 │ 图  │    ...  │ (高=相关)│
...    └─────┘    1     └──────────┘

每个 (lag, band) 点显示：
该频带在 lag 延迟处与自身的相关系数
```

---

## 5. 跨域对齐 (Cross-Domain Consistency)

所有三种表示在 XRF55 和 WiSe4Car 之间保持一致：

### 对齐清单
| 特征 | XRF55 | WiSe4Car | 对齐状态 |
|------|-------|----------|--------|
| 1D 时间长度 | T=256 | T=256 | ✅ |
| 1D 频带数 | F=16 | F=16 | ✅ |
| STFT 参数 | 同下 | 同下 | ✅ |
| STFT nperseg | 64 | 64 | ✅ |
| STFT nfft | 128 | 128 | ✅ |
| ACF max_lag | 64 | 64 | ✅ |
| ACF 归一化 | lag-0 norm | lag-0 norm | ✅ |
| 标签集合 | 6 类 | 6 类 | ✅ |
| 标签顺序 | [Sitting, Reaching, Turning, Bending, Waving, Using Phone] | 同 | ✅ |

### 优势
1. **直接迁移**: 同一模型可在两个域上训练
2. **公平对标**: 表示形式一致，便于准确性比对
3. **理论保证**: 统一的预处理确保概率分布兼容

---

## 6. 生成的可视化文件

### 1. 三表示对比图 (`csi_representations_{label}.png/pdf`)

包含 3 个并排子图：

**(a) 1D CSI Sequence** — [256, 16] 热力图
- X轴: 时间步 (0~256)
- Y轴: 频带 (0~16)
- 颜色: 振幅 (RdBu_r 配色，±3σ 范围)

**(b) STFT (频带平均)** — [65频率 × 13时间帧]
- X轴: STFT 时间帧数 (13)
- Y轴: 频率箱 (0~65)
- 颜色: 对数功率谱 (viridis 配色)
- 黄色区域 = 高能频率

**(c) ACF (逐频带)** — [65延迟 × 16频带]
- X轴: 频带 (0~16)
- Y轴: 相关延迟 (0~64 lag)
- 颜色: 归一化自相关系数 (coolwarm 配色，-1~+1)

### 2. 逐频带细节图 (`per_band_analysis_{label}.png/pdf`)

6 个子图：

**顶行**: 4 个选定频带的 STFT 热力图
**底行**: 对应频带的 ACF 曲线

---

## 7. 快速开始

### 生成单个样本对比
```bash
cd Experiment_Mainline/04_Preprocessing/scripts
python visualize_csi_representations.py \
    --domain xrf55 \
    --sample_idx 0 \
    --output_dir ../results
```

### 生成带详细分析的对比
```bash
python visualize_csi_representations.py \
    --domain wise4car \
    --sample_idx 10 \
    --output_dir ../results \
    --per_band
```

### 输出文件结构
```
results/
├── csi_representations_Using Phone.png     # PNG 400dpi
├── csi_representations_Using Phone.pdf     # PDF 矢量
├── per_band_analysis_Using Phone.png       # 详细分析 (可选)
└── per_band_analysis_Using Phone.pdf       # PDF 版本 (可选)
```

---

## 8. 建议用途

### 论文中的用法
- **方法论章节**: 使用 (a)(b)(c) 三图展示数据处理管道
- **实验结果**: STFT 结果用于 CNN 模型，ACF 结果用于周期性分析
- **附录**: per_band_analysis 作为补充材料

### 模型开发
1. 从 1D 序列开始（最简单基线）
2. 升级到 STFT（如需要时频信息）
3. 融合 ACF（如需要周期性）
4. 多表示融合（三种同时用）

---

## 9. 相关文件

- `preprocess_stft_acf.py` — 全数据集批量生成脚本
- `stft_acf_preprocessing_design.md` — 设计决策详解
- `preprocessing_summary.md` — WiSe4Car / XRF55 原始预处理说明

---

## 更新日期

- **2026-05-04**: 初始文档 + 三表示可视化脚本
- **源数据**: Transfer_Learning/Data/XRF55, Transfer_Learning/Data/Wise4Car
