import numpy as np
import matplotlib.pyplot as plt
import os

models = ['MLP','LeNet','RNN','BiLSTM','CNN+GRU']
results_dir = 'quick_results'

train_accs = []
test_accs = []

for m in models:
    path = os.path.join(results_dir, f'results_{m}.npz')
    if not os.path.exists(path):
        print('Missing', path)
        continue
    data = np.load(path)
    # take last test acc2 as representative
    test_acc = data['sup_test_acc2'][-1] if data['sup_test_acc2'].size>0 else 0
    train_acc = data['sup_train_acc2'][-1] if data['sup_train_acc2'].size>0 else 0
    train_accs.append(train_acc)
    test_accs.append(test_acc)

# fallback lengths
if len(train_accs) != len(models):
    print('Some models missing, plotting available ones')
    models = models[:len(train_accs)]

x = range(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(10,5))
ax.bar([i-width/2 for i in x], train_accs, width, label='Train')
ax.bar([i+width/2 for i in x], test_accs, width, label='Test')
ax.set_xticks(list(x))
ax.set_xticklabels(models)
ax.set_ylabel('Accuracy (%)')
ax.set_title('Quick-run Results (1 self-epoch, 2 sup-epochs)')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir,'quick_comparison.png'), dpi=200)
print('Saved', os.path.join(results_dir,'quick_comparison.png'))
