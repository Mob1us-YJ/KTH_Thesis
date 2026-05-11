import numpy as np
import matplotlib.pyplot as plt
import os

path = 'quick_results/results_MLP.npz'
if not os.path.exists(path):
    raise SystemExit('Missing results file: ' + path)

data = np.load(path)
unsup = data['unsup_epoch_losses'] if 'unsup_epoch_losses' in data else None
unsup_kl = data['unsup_kl'] if 'unsup_kl' in data else None
unsup_eh = data['unsup_eh'] if 'unsup_eh' in data else None
unsup_he = data['unsup_he'] if 'unsup_he' in data else None
unsup_kde = data['unsup_kde'] if 'unsup_kde' in data else None

sup_loss = data['sup_epoch_losses'] if 'sup_epoch_losses' in data else None
sup_train_acc1 = data['sup_train_acc1'] if 'sup_train_acc1' in data else None
sup_train_acc2 = data['sup_train_acc2'] if 'sup_train_acc2' in data else None
sup_test_acc1 = data['sup_test_acc1'] if 'sup_test_acc1' in data else None
sup_test_acc2 = data['sup_test_acc2'] if 'sup_test_acc2' in data else None

fig, axes = plt.subplots(1,2,figsize=(14,5))
# Left: unsupervised losses
ax = axes[0]
if unsup is not None and unsup.size>0:
    ax.plot(np.arange(1, len(unsup)+1), unsup, marker='o', label='Self-supervised total loss')
if unsup_kde is not None and unsup_kde.size>0:
    ax.plot(np.arange(1, len(unsup_kde)+1), unsup_kde, marker='x', label='KDE loss')
ax.set_xlabel('Self epochs')
ax.set_ylabel('Loss')
ax.set_title('Self-supervised losses (MLP)')
ax.grid(True, alpha=0.3)
ax.legend()

# Right: supervised loss and accuracies
ax2 = axes[1]
if sup_loss is not None and sup_loss.size>0:
    ax2.plot(np.arange(1, len(sup_loss)+1), sup_loss, color='tab:blue', label='Supervised train loss')
ax2.set_xlabel('Super epochs')
ax2.set_ylabel('Loss', color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')

ax3 = ax2.twinx()
if sup_train_acc2 is not None and sup_train_acc2.size>0:
    ax3.plot(np.arange(1, len(sup_train_acc2)+1), sup_train_acc2, color='tab:green', linestyle='--', marker='o', label='Train acc (head2)')
if sup_test_acc2 is not None and sup_test_acc2.size>0:
    ax3.plot(np.arange(1, len(sup_test_acc2)+1), sup_test_acc2, color='tab:red', linestyle='--', marker='s', label='Test acc (head2)')
ax3.set_ylabel('Accuracy (%)', color='tab:red')
ax3.tick_params(axis='y', labelcolor='tab:red')

lines, labels = ax2.get_legend_handles_labels()
lines2, labels2 = ax3.get_legend_handles_labels()
ax2.legend(lines+lines2, labels+labels2, loc='upper left')
ax2.set_title('Supervised training (MLP)')
ax2.grid(True, alpha=0.2)

out = 'quick_results/results_MLP_plots.png'
plt.tight_layout()
plt.savefig(out, dpi=200)
print('Saved', out)
