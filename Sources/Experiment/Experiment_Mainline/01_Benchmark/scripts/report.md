**实验方法报告**

- **作者**: （填写）
- **日期**: 2026-02-23

**摘要**:
- 本次实验评估“先无监督预训练（NTU‑Fi）→再有监督线性评估（NTU‑Fi‑HumanID）”的迁移性能。比较模型：MLP、LeNet（CNN）、RNN、BiLSTM、CNN+GRU。关键输出包括训练/测试 accuracy、train–test gap 以及收敛曲线。

**数据与划分**:
- 无监督预训练数据集：NTU‑Fi_HAR 的 `train_amp` 与 `test_amp`（在训练脚本中两者合并为 unsupervised 训练集）。
- 监督微调训练集（linear eval）：NTU‑Fi‑HumanID 的 `test_amp`（用于训练线性分类头）。
- 最终评估测试集：NTU‑Fi‑HumanID 的 `train_amp`（作为测试集评估）。
- 注意：脚本默认 `root="./Data/"`，请确认数据实际路径或在运行时调整 root。

**预处理与增强**:
- 数据由 `dataset.py` 与 `util.py` 加载为 `CSI_Dataset`，转为 PyTorch `DataLoader`（batch_size=64）。
- 主要数据增强：基于 `gaussian_noise(x, eps)` 生成两视图（x1, x2），eps 在训练中随机采样以生成扰动视图。

**模型**:
- 使用并行封装的模型变体（例如 `CNN_GRU_Parrallel`）以支持 self-supervised 双视图输入与两路分类 head。模型定义见 `self_supervised_model.py` 与各数据集模型文件（`NTU_Fi_model.py`, `UT_HAR_model.py`）。

**训练协议**:
- 无监督阶段（Representation Learning）: 训练 encoder + projector，损失为 `EntLoss`（KL + EH + HE + KDE 组合），优化器 `AdamW(model.parameters(), lr=1e-3, weight_decay=args.weight_decay)`。本次完整运行使用 `--self-epochs 100`。
- 有监督阶段（Linear Evaluation）: 只训练 `model.classifier`（encoder 冻结），损失为两个 head 的交叉熵之和，优化器 `Adam(model.classifier.parameters(), lr=1e-3, weight_decay=1e-5)`，本次使用 `--super-epochs 300`。

**关键超参数**:
- learning_rate (self/super): 1e-3
- batch_size: 64
- weight_decay: self 阶段 `args.weight_decay`（默认 1.5e-6），super 阶段 1e-5
- tau: 1.0, EPS: 1e-5, lam1: 0.0, lam2: 1.0
- self-epochs: 100（完整），super-epochs: 300（完整）

**损失与记录**:
- `EntLoss` 记录项: `kl`, `eh`, `he`, `kde`，最终组合 `final-kde` 用于反向传播。
- 每轮记录并保存到 `quick_results/results_<model>.npz`：包含 `unsup_epoch_losses`, `unsup_kl`, `sup_epoch_losses`, `sup_train_acc1/2`, `sup_test_acc1/2` 等。

**评估指标**:
- 主要指标：Train / Test Top‑1 accuracy（两个 head 分别记录）；报告以最后一轮 supervised 测试 acc 为主对比。
- 同时给出 train–test gap 与训练收敛曲线（loss / acc 曲线）。

**复现命令**:
```
# 快速验证（1 自监督 + 2 有监督）
python self_supervised.py --model MLP --self-epochs 1 --super-epochs 2

# 完整迁移实验（示例）
python self_supervised.py --model "CNN+GRU" --self-epochs 100 --super-epochs 300

# 生成汇总对比图（已包含所有模型）
python BenchTest/plot_migration_comparison.py
```

**已生成的产物（路径）**:
- 快速/完整结果：`quick_results/results_<model>.npz`（例如 `quick_results/results_CNN+GRU.npz`）
- 总体比较图：`BenchTest/migration_comparison.png`
- 模型/详细图：`quick_results/results_LeNet_plots.png` 等（若存在）

**已知问题与诊断建议（重点：CNN+GRU）**:
- 可能原因：
  - CNN→GRU 的时间/通道维度顺序不匹配（batch_first、序列维度误用）；
  - GRU 配置（hidden_size 太小 / 单向而需双向）或 CNN 下采样过强导致时间分辨率丢失；
  - 训练策略（只训练 head，但 encoder 未学到有效表示）；
  - 数据路径或预处理与模型期望不一致。
- 排查步骤：
  1. 打印 CNN 输出与送入 GRU 前后的 `tensor.shape` 以确认维度（batch, seq, feat）是否匹配；
  2. 在小 batch（1–4）上做前向+反向检查，确认梯度流；
  3. 尝试短跑实验（10–20 epoch）：开启双向 GRU / 增大 hidden_size / 减少下采样，或仅训练 encoder 看表示质量；
  4. 绘制 CNN+GRU 的训练/验证 loss 与 acc 曲线以判断欠拟合/不收敛/振荡。

**报告撰写建议（用于提交/幻灯片）**:
- 页面顺序：摘要 → 数据说明（含样本数）→ 模型与超参 → 关键图表（总体比较 + 每模型收敛曲线）→ 问题诊断（CNN+GRU）→ 结论与下一步。

**联系方式 / 下一步**:
- 我可以把此文件另存为 PDF，或把诊断步骤 1–4 自动化为脚本并运行（需你选择）。
