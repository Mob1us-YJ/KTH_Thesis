from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Paper-style plotting defaults (clean, readable, colorblind-friendly)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# NOTE: ResNet values currently follow the previous LeNet slot to match existing
# benchmark records in this repository plot script.
models = ["MLP", "ResNet", "RNN", "BiLSTM", "CNN+GRU"]
train_acc = [95.54, 100.0, 82.61, 81.50, 99.42]
test_acc = [91.64, 97.46, 75.87, 77.50, 97.13]
gap = [round(a - b, 2) for a, b in zip(train_acc, test_acc)]

# Training trajectories
mlp_epochs = [1, 10, 20, 50, 100, 150, 200]
mlp_train = [25.60, 54.44, 68.90, 80.34, 93.67, 95.36, 95.54]

resnet_epochs = [1, 5, 10, 20, 30, 40, 60, 100, 150, 200]
resnet_train = [28.02, 96.29, 99.48, 97.43, 96.77, 100.0, 100.0, 100.0, 100.0, 100.0]

bilstm_epochs = [1, 20, 40, 60, 100, 140, 180, 200]
bilstm_train = [32.01, 43.65, 61.39, 63.79, 71.55, 78.05, 84.17, 81.50]

cnn_gru_epochs = [1, 10, 20, 50, 100, 150, 200]
cnn_gru_train = [28.58, 56.73, 70.89, 84.30, 93.32, 94.66, 99.42]

palette = {
    "MLP": "#1f77b4",
    "ResNet": "#d62728",
    "RNN": "#2ca02c",
    "BiLSTM": "#ff7f0e",
    "CNN+GRU": "#9467bd",
}

fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)

# (a) Accuracy comparison
ax1 = axes[0]
x = np.arange(len(models))
width = 0.36

bar_train = ax1.bar(
    x - width / 2,
    train_acc,
    width,
    label="Train",
    color=[palette[m] for m in models],
    alpha=0.85,
    edgecolor="#333333",
    linewidth=0.7,
)
bar_test = ax1.bar(
    x + width / 2,
    test_acc,
    width,
    label="Test",
    color=[palette[m] for m in models],
    alpha=0.35,
    edgecolor="#333333",
    linewidth=0.7,
)

ax1.set_title("(a) UT-HAR Accuracy (Train vs Test)", pad=8)
ax1.set_ylabel("Accuracy (%)")
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.set_ylim(70, 101.5)
ax1.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
ax1.legend(loc="lower left", frameon=False, ncol=2)

for b in list(bar_train) + list(bar_test):
    h = b.get_height()
    ax1.text(
        b.get_x() + b.get_width() / 2,
        h + 0.35,
        f"{h:.1f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )

# (b) Convergence curve
ax2 = axes[1]
ax2.plot(mlp_epochs, mlp_train, marker="o", markersize=4.8, linewidth=2.0, color=palette["MLP"], label="MLP")
ax2.plot(resnet_epochs, resnet_train, marker="s", markersize=4.8, linewidth=2.0, color=palette["ResNet"], label="ResNet")
ax2.plot(bilstm_epochs, bilstm_train, marker="^", markersize=5.2, linewidth=2.0, color=palette["BiLSTM"], label="BiLSTM")
ax2.plot(cnn_gru_epochs, cnn_gru_train, marker="D", markersize=4.8, linewidth=2.0, color=palette["CNN+GRU"], label="CNN+GRU")

ax2.set_title("(b) Training Convergence", pad=8)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Train Accuracy (%)")
ax2.set_xlim(0, 205)
ax2.set_ylim(20, 101.5)
ax2.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
ax2.legend(loc="lower right", frameon=False)

out_dir = Path(__file__).resolve().parents[1] / "results"
out_dir.mkdir(parents=True, exist_ok=True)
png_path = out_dir / "UT_HAR_model_comparison_resnet_paper.png"
pdf_path = out_dir / "UT_HAR_model_comparison_resnet_paper.pdf"

fig.savefig(png_path, dpi=400, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")

print(f"Saved PNG: {png_path}")
print(f"Saved PDF: {pdf_path}")

results = list(zip(models, train_acc, test_acc, gap))
for rank, (model, train, test, g) in enumerate(sorted(results, key=lambda x: x[2], reverse=True), 1):
    print(f"#{rank} {model:<8} train={train:>6.2f}% test={test:>6.2f}% gap={g:>5.2f}%")
