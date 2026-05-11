# XRF55 预处理后模型训练摘要

数据与输入格式
- 数据源：Transfer_Learning/Data/XRF55 下的 `XRF55_features.npy`（shape [N, 256, 16]，幅度特征），`XRF55_labels.npy`（6 类：Sitting, Reaching, Turning, Bending, Waving, Using Phone），`XRF55_train_indices.npy`/`XRF55_val_indices.npy` 用于划分。
- 预处理：融合多链路/多 RX，时间对齐到 4.6s，重采样 256 步，band pooling 到 16 维，线性去趋势 + 逐特征 z-score。
- 域标识：`XRF55_domains.npy` 全 0（源域）。

模型（`train_xrf55_encoder.py` 当前版本）
- Backbone：时间卷积编码器（4 层 1D Conv + BN + ReLU + GAP）。
  - 通道：`width` → `2*width` → `4*width` → `8*width`，默认 width=64（即 64-128-256-512）。
  - 输入通道：16（频率/子载波维）。
  - Dropout：可调（默认 0.3），放在 GAP 后、全连接前。
- 分类头：Linear(8*width → 6 类)。
- 损失：CrossEntropyLoss，带类权重（按训练集频次倒数）与可调 label smoothing。
- 优化器：Adam（lr 默认 1e-3，weight_decay 可调）。
- 学习率调度：StepLR(step=10, gamma=0.5) 或 CosineAnnealingLR(T_max=epochs)，默认 cosine。
- 早停：patience 可调，监控 val acc。
- 可选 mixup：beta(alpha, alpha)。
- 可选 WeightedRandomSampler：按类频次倒数采样。

已跑实验与结果（验证集最佳准确率）
- preproc_balanced_dropout：sampler 开启，ls=0.05，dropout=0.3，wd=1e-4，epochs=80，patience=15，scheduler=cosine，width=64 → best val acc ≈ 0.6936。
- preproc_nosampler_lowwd：sampler 关，ls=0.02，dropout=0.4，wd=1e-5，epochs=80，patience=20，scheduler=cosine，width=64 → best val acc ≈ 0.7694。
- preproc_mixup_cosine：sampler 关，mixup_alpha=0.2，ls=0.02，dropout=0.35，wd=1e-5，epochs=120，patience=25，scheduler=cosine，width=64 → best val acc ≈ 0.7037。
- preproc_cosine_nomixup：sampler 关，ls=0.04，dropout=0.35，wd=1e-4，epochs=120，patience=20，scheduler=cosine，width=64 → best val acc ≈ 0.8232。
- preproc_cosine_wide：sampler 关，ls=0.04，dropout=0.35，wd=1e-4，epochs=140，patience=25，scheduler=cosine，width=96 → best val acc ≈ 0.8552。

可视化
- 验证精度柱状图：`Transfer_Learning/result/xrf55_val_acc_summary.png`（手工记录上述 run 的 best val acc 绘制）。

备注
- 所有 run 均使用 CPU（pin_memory 警告提示无加速器）。
- 训练/验证划分来自预处理生成的索引；未改动数据划分。
- 当前最高验证准确率来自更宽通道 + cosine 调度（preproc_cosine_wide）。
