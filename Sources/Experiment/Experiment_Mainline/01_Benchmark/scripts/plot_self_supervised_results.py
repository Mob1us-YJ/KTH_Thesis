import os
import numpy as np
import matplotlib.pyplot as plt


def load_results(path):
    try:
        data = np.load(path, allow_pickle=True)
        return dict(data)
    except Exception:
        return {}


def get_accs(data):
    # return (train_curve, test_curve, final_train, final_test)
    train1 = data.get('sup_train_acc1')
    train2 = data.get('sup_train_acc2')
    test1 = data.get('sup_test_acc1')
    test2 = data.get('sup_test_acc2')

    # prefer paired arrays
    if train1 is None and train2 is None:
        # try other keys
        for k in ('train_accs', 'train_acc'):
            if k in data:
                arr = np.array(data[k])
                return arr, arr, float(arr[-1]) if arr.size>0 else None, float(arr[-1]) if arr.size>0 else None
        return None, None, None, None

    train1 = np.array(train1) if train1 is not None else None
    train2 = np.array(train2) if train2 is not None else None
    test1 = np.array(test1) if test1 is not None else None
    test2 = np.array(test2) if test2 is not None else None

    # build train curve as average if both exist
    if train1 is not None and train2 is not None:
        train_curve = (train1 + train2) / 2.0
    else:
        train_curve = train1 if train1 is not None else train2

    if test1 is not None and test2 is not None:
        test_curve = (test1 + test2) / 2.0
    else:
        test_curve = test1 if test1 is not None else test2

    final_train = float(train_curve[-1]) if train_curve is not None and train_curve.size>0 else None
    final_test = float(test_curve[-1]) if test_curve is not None and test_curve.size>0 else None
    return train_curve, test_curve, final_train, final_test


def main():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    qdir = os.path.join(base, 'quick_results')

    models = ['MLP', 'LeNet\n(CNN)', 'RNN', 'BiLSTM\n(Fixed)', 'CNN+GRU']
    file_map = {
        'MLP': 'results_MLP.npz',
        'LeNet\n(CNN)': 'results_LeNet.npz',
        'RNN': 'results_RNN.npz',
        'BiLSTM\n(Fixed)': 'results_BiLSTM.npz',
        'CNN+GRU': 'results_CNN+GRU.npz'
    }

    final_trains = []
    final_tests = []
    curves = {}

    for m in models:
        fname = file_map.get(m)
        p = os.path.join(qdir, fname)
        if not os.path.exists(p):
            print(f'Warning: missing results for {m}: {p}')
            final_trains.append(0)
            final_tests.append(0)
            continue
        data = load_results(p)
        train_curve, test_curve, final_train, final_test = get_accs(data)
        final_trains.append(final_train if final_train is not None else 0)
        final_tests.append(final_test if final_test is not None else 0)
        if train_curve is not None:
            curves[m] = train_curve

    # Create figure with matching style
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Model Performance Comparison on Self Supervised Learning', fontsize=16, fontweight='bold')
    axbar, axcurve = axes

    x = np.arange(len(models))
    width = 0.35
    colors_train = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    colors_test = ['#5dade2', '#ec7063', '#58d68d', '#f5b041', '#af7ac5']

    bars1 = axbar.bar(x - width/2, final_trains, width, label='Train Accuracy', color=colors_train, alpha=0.9, edgecolor='black', linewidth=1.5)
    bars2 = axbar.bar(x + width/2, final_tests, width, label='Test Accuracy', color=colors_test, alpha=0.9, edgecolor='black', linewidth=1.5)
    axbar.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    axbar.set_title('Train vs Test Accuracy Comparison', fontsize=14, fontweight='bold')
    axbar.set_xticks(x)
    axbar.set_xticklabels(models, fontsize=11)
    axbar.set_ylim(0, 105)
    axbar.legend(fontsize=11, loc='upper right')
    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                axbar.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Plot training convergence curves with styled markers
    marker_map = {
        'MLP': ('o', '#3498db'),
        'LeNet\n(CNN)': ('s', '#e74c3c'),
        'RNN': ('^', '#2ecc71'),
        'BiLSTM\n(Fixed)': ('^', '#f39c12'),
        'CNN+GRU': ('d', '#9b59b6')
    }
    for name in models:
        if name in curves:
            curve = curves[name]
            epochs = np.arange(1, len(curve) + 1)
            m, c = marker_map.get(name, ('o', '#333333'))
            axcurve.plot(epochs, curve, marker=m, linestyle='-', linewidth=2.5, markersize=7, markerfacecolor='white', markeredgewidth=1.8, color=c, label=name)

    axcurve.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    axcurve.set_ylabel('Training Accuracy (%)', fontsize=13, fontweight='bold')
    axcurve.set_title('Training Convergence Speed', fontsize=14, fontweight='bold')
    axcurve.set_ylim(0, 105)
    axcurve.grid(True, linestyle='--', alpha=0.3)
    axcurve.legend(fontsize=11, loc='lower right')

    plt.tight_layout()
    out = os.path.join('BenchTest', 'self_supervised_comparison.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
