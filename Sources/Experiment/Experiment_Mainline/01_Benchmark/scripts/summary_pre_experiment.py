"""Generate a single summary figure for pre-experiments.

Left bars: Supervised on NTU-Fi HAR (test accuracy).
Right bars: Self-supervised pretrain on NTU-Fi HAR + linear eval on NTU-Fi HumanID (test accuracy head1).
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 13

root = Path(__file__).parent
har_json = root / "ntufi_results.json"
transfer_dir = root / "quick_results"
models = ["MLP", "LeNet", "RNN", "BiLSTM", "CNN+GRU"]

# supervised HAR test accuracy
with har_json.open() as f:
    har_results = json.load(f)

test_har = [har_results[m]["test_acc"] for m in models]

# transfer (self-supervised then linear eval) test_acc1
transfer_acc = []
for m in models:
    data = np.load(transfer_dir / f"results_{m}.npz")
    acc = float(data["sup_test_acc1"][-1]) if "sup_test_acc1" in data else np.nan
    transfer_acc.append(acc)

x = np.arange(len(models))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 5))

bar1 = ax.bar(x - width / 2, test_har, width, color="#4e79a7", label="Supervised on HAR")
bar2 = ax.bar(x + width / 2, transfer_acc, width, color="#f28e2b", label="Self-sup → HumanID")

ax.set_ylabel("Test Accuracy (%)")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15)
ax.set_ylim(0, 105)
ax.legend(loc="upper right")
ax.grid(axis="y", linestyle="--", alpha=0.3)

# add value labels
for bars in (bar1, bar2):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 1, f"{h:.1f}%", ha="center", va="bottom", fontsize=11)

fig.tight_layout()
out_path = root / "pre_experiment_summary.png"
fig.savefig(out_path, dpi=200)
print("Saved", out_path)
