# 04 Preprocessing 模块总结

## 概述

新创建的 `04_Preprocessing` 模块是对前三个主线的补充，重点展示了 CSI 数据在三种不同表示形式中的特性对比：**1D Sequence、2D STFT、2D ACF**。

## 核心功能

### 1. 三种表示形式对比

| 表示 | 形状 | 用途 | 优势 |
|-----|------|------|------|
| **1D CSI** | [256, 16] | 基线模型、特征融合 | 直观、完整、低延迟 |
| **STFT** | [16, 65, 13] | CNN/RNN 特征、时频分析 | 捕捉谱变化、易可视化 |
| **ACF** | [65, 16] | 周期性识别、速度估计 | 鲁棒噪声、可解释 |

### 2. 统一预处理前端

所有三种表示共享相同的初始处理阶段：
- 时间轴重采样 → T=256 (≈4.6 秒)
- 频带池化 → F=16 (均匀分布的子载波组)
- 去趋势 + z-score 标准化 + ±6σ 裁剪

这确保了 **XRF55 和 WiSe4Car 之间的完全一致性**。

## 文件结构

```
04_Preprocessing/
├── README.md                           # 快速开始指南
├── 03_CSI_Representations_Guide.md    # 详细技术文档
├── scripts/
│   ├── visualize_csi_representations.py  # 主可视化脚本 (≈400 行)
│   ├── preprocess_stft_acf.py            # 全数据集转换脚本 (复制)
│   ├── preprocessing_summary.md          # 预处理总结 (复制)
│   └── stft_acf_preprocessing_design.md   # 设计文档 (复制)
└── results/
    ├── csi_representations_Using Phone.png     # 三表示对比 (400dpi)
    ├── csi_representations_Using Phone.pdf     # 矢量格式
    ├── per_band_analysis_Using Phone.png       # 逐频带分析 (400dpi)
    └── per_band_analysis_Using Phone.pdf       # 矢量格式
```

## 生成的可视化

### 图表 A: 三表示对比 (`csi_representations_*.png/pdf`)

**面板 (a) — 1D CSI Sequence**
- 纵轴: 16 个频带
- 横轴: 256 个时间步
- 热力图显示原始 CSI 幅度值
- 颜色条: RdBu_r (红=正, 蓝=负)

**面板 (b) — STFT (频带平均)**
- 纵轴: 65 个频率箱 (FFT 输出)
- 横轴: 13 个时间帧 (hop_length=16)
- 颜色表示对数功率谱强度
- 黄色区域 = 高能频率成分

**面板 (c) — ACF (逐频带)**
- 纵轴: 65 个相关延迟 (lag 0~64)
- 横轴: 16 个频带
- 蓝色 = 负相关, 红色 = 正相关
- 体现了不同频带的时间依赖性

### 图表 B: 逐频带细节分析 (`per_band_analysis_*.png/pdf`)

**顶行 (4 个热力图)**: 
- 频带 0, 5, 10, 15 的 STFT 时频谱
- 演示跨频带的能量分布差异

**底行 (4 条曲线)**:
- 对应频带的 ACF 函数
- 蓝色柱状显示相关系数变化
- 周期性峰值对应运动节奏

## 关键参数

### STFT 配置
- Window: Hann
- nperseg: 64 (分析窗口)
- noverlap: 48 (重叠长度)
- hop_length: 16 (窗口步长)
- nfft: 128 (FFT 点数)
- Output: 13 时间帧 × 65 频率箱

### ACF 配置
- max_lag: 64 (最大延迟)
- normalize_by_lag0: True (相对于方差归一化)
- 时间尺度: lag_0~lag_64 ≈ 0~1.14 秒

## 快速使用

### 生成对比图 (基础)
```bash
python scripts/visualize_csi_representations.py \
    --domain xrf55 \
    --sample_idx 0 \
    --output_dir ./results
```

### 生成详细分析 (带逐频带图)
```bash
python scripts/visualize_csi_representations.py \
    --domain wise4car \
    --sample_idx 10 \
    --output_dir ./results \
    --per_band
```

### 支持的选项
| 参数 | 可选值 | 说明 |
|------|--------|------|
| `--domain` | `xrf55` / `wise4car` | 数据源 |
| `--sample_idx` | 整数 | 样本索引 |
| `--output_dir` | 路径 | 输出目录 |
| `--per_band` | 标志 | 启用逐频带分析 |

## 数据来源

- **1D 预处理数据**: `Transfer_Learning/Data/XRF55/` 和 `Transfer_Learning/Data/Wise4Car/`
- **预处理脚本**: `Transfer_Learning/preprocess_stft_acf.py`
- **设计文档**: `Transfer_Learning/result/stft_acf_preprocessing_design.md`

## 学术应用

### 适用场景

1. **论文方法论章节**
   - 用 (a)(b)(c) 三图展示完整预处理管道
   - 说明为什么需要多种表示

2. **实验结果部分**
   - 对比不同模型在三种表示上的性能
   - STFT 用于 CNN 模型，ACF 用于序列模型

3. **附录或补充材料**
   - per_band_analysis 展示频带间的差异
   - 验证预处理的有效性

### 会议投稿标准

✅ **符合要求**:
- 双面板布局 (12.8×4.8 英寸, NeurIPS/ICLR 标准)
- 400 dpi PNG 用于呈现
- 矢量 PDF 用于排版
- 清晰标注 (a)(b)(c)
- 色盲友好配色 (RdBu_r, viridis, coolwarm)

## 与其他模块的关系

```
01_Benchmark
    ↓ (模型在这些表示上训练/测试)
    
04_Preprocessing ← 数据管道
    ↓ (提供标准化特征)
    
02_Annotation & 03_XRF55
    ↓ (标注/预处理结果)
    
Transfer_Learning/
    ↓ (使用统一的数据格式)
    
实验结果
```

## 扩展建议

### 可增加的可视化
1. **时间序列曲线**: 某个频带随时间的波形
2. **频谱对比**: 不同类别的平均 STFT 功率谱
3. **ACF 峰值分析**: 自动检测周期性特征
4. **多类别网格**: 6 个类别各 1 个样本的 3×6 对比

### 可集成的功能
1. 交互式参数调整 (jupyter/streamlit)
2. 频率特征检测 (主导频率、谐波)
3. 跨域表示距离度量 (MMD, CORAL)
4. 类别特定的统计汇总

## 文件大小

| 文件 | 大小 | 格式 |
|------|------|------|
| visualize_csi_representations.py | ≈7 KB | Python |
| csi_representations_*.png | ≈300-400 KB | 400dpi |
| csi_representations_*.pdf | ≈50-100 KB | 矢量 |
| per_band_analysis_*.png | ≈400 KB | 400dpi |
| per_band_analysis_*.pdf | ≈100 KB | 矢量 |
| 03_CSI_Representations_Guide.md | ≈12 KB | 技术文档 |

## 更新日志

- **2026-05-04**: 
  - 创建 04_Preprocessing 模块
  - 编写三表示对比脚本 (≈400 行)
  - 生成 XRF55/WiSe4Car 样本对比图
  - 编写详细技术文档 (≈400 行)
  - 生成逐频带分析图

## 备注

所有脚本已优化以支持：
- ✅ 论文发表级图表质量 (400 dpi PNG + PDF vector)
- ✅ 跨域一致性 (XRF55 和 WiSe4Car 相同参数)
- ✅ 易扩展性 (脚本参数化，易修改表示方式)
- ✅ 可复现性 (确定性流程，无随机性)

---

**下一步**: 可选为其他类别样本 (Sitting, Turning, etc.) 生成对比图，或集成至论文投稿。
