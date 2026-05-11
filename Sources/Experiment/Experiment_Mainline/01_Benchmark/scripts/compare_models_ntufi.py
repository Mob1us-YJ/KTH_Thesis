"""
NTU-Fi HAR 数据集模型对比实验
自动运行 MLP, LeNet, RNN, BiLSTM, CNN+GRU 五个模型并生成对比图
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from util import load_data_n_model
import json
import os

# 设置字体
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

def train_and_evaluate(model, train_loader, test_loader, num_epochs, learning_rate, device):
    """训练模型并返回训练曲线和最终准确率"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    train_acc_history = []
    epoch_checkpoints = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        
        for data in train_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device).long()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.to(device).float()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * inputs.size(0)
            predict_y = torch.argmax(outputs, dim=1).to(device)
            epoch_accuracy += (predict_y == labels).sum().item() / labels.size(0)
        
        epoch_accuracy = epoch_accuracy / len(train_loader) * 100
        
        # 记录检查点
        checkpoint_epochs = [1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
        if (epoch + 1) in checkpoint_epochs or (epoch + 1) == num_epochs:
            train_acc_history.append(epoch_accuracy)
            epoch_checkpoints.append(epoch + 1)
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {epoch_accuracy:.2f}%')
    
    # 测试
    model.eval()
    test_acc = 0
    final_train_acc = epoch_accuracy
    
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device).long()
            
            outputs = model(inputs)
            predict_y = torch.argmax(outputs, dim=1)
            test_acc += (predict_y == labels).sum().item()
    
    test_acc = test_acc / len(test_loader.dataset) * 100
    print(f'Final Train Accuracy: {final_train_acc:.2f}%, Test Accuracy: {test_acc:.2f}%')
    
    return {
        'train_acc': final_train_acc,
        'test_acc': test_acc,
        'train_history': train_acc_history,
        'epochs': epoch_checkpoints
    }


def run_all_experiments():
    """运行所有模型实验"""
    root = './Data/'
    dataset_name = 'NTU-Fi_HAR'
    models_to_test = ['MLP', 'LeNet', 'RNN', 'BiLSTM', 'CNN+GRU']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    results = {}
    
    for model_name in models_to_test:
        print(f"\n{'='*50}")
        print(f"Training {model_name} on {dataset_name}")
        print('='*50)
        
        train_loader, test_loader, model, train_epoch = load_data_n_model(dataset_name, model_name, root)
        
        result = train_and_evaluate(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            num_epochs=train_epoch,
            learning_rate=1e-3,
            device=device
        )
        
        results[model_name] = result
    
    # 保存结果
    with open('ntufi_results.json', 'w') as f:
        json.dump({k: {'train_acc': v['train_acc'], 'test_acc': v['test_acc'], 
                       'train_history': v['train_history'], 'epochs': v['epochs']} 
                   for k, v in results.items()}, f, indent=2)
    
    return results


def plot_comparison(results):
    """生成对比图"""
    models = ['MLP', 'LeNet\n(CNN)', 'RNN', 'BiLSTM', 'CNN+GRU']
    model_keys = ['MLP', 'LeNet', 'RNN', 'BiLSTM', 'CNN+GRU']
    
    train_acc = [results[k]['train_acc'] for k in model_keys]
    test_acc = [results[k]['test_acc'] for k in model_keys]
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Model Performance Comparison on NTU-Fi HAR Dataset', fontsize=16, fontweight='bold')
    
    # 1. 训练vs测试准确率对比（柱状图）
    ax1 = axes[0]
    x = np.arange(len(models))
    width = 0.35
    colors_train = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    colors_test = ['#5dade2', '#ec7063', '#58d68d', '#f5b041', '#af7ac5']
    
    bars1 = ax1.bar(x - width/2, train_acc, width, label='Train Accuracy',
                    color=colors_train, alpha=0.9, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, test_acc, width, label='Test Accuracy',
                    color=colors_test, alpha=0.9, edgecolor='black', linewidth=1.5)
    
    ax1.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax1.set_title('Train vs Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=11)
    ax1.set_ylim([0, 105])
    ax1.legend(fontsize=11, loc='upper right')
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 2. 训练收敛曲线
    ax2 = axes[1]
    
    colors = {'MLP': '#3498db', 'LeNet': '#e74c3c', 'RNN': '#2ecc71', 
              'BiLSTM': '#f39c12', 'CNN+GRU': '#9b59b6'}
    markers = {'MLP': 'o', 'LeNet': 's', 'RNN': '^', 'BiLSTM': 'D', 'CNN+GRU': 'd'}
    labels = {'MLP': 'MLP', 'LeNet': 'LeNet (CNN)', 'RNN': 'RNN', 
              'BiLSTM': 'BiLSTM', 'CNN+GRU': 'CNN+GRU'}
    
    for model_name in model_keys:
        epochs = results[model_name]['epochs']
        history = results[model_name]['train_history']
        ax2.plot(epochs, history, f'{markers[model_name]}-', label=labels[model_name],
                 color=colors[model_name], linewidth=2.5, markersize=7,
                 markerfacecolor='white', markeredgewidth=2)
    
    ax2.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Training Accuracy (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Training Convergence Speed', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='lower right')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig('ntufi_model_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✅ Visualization saved: ntufi_model_comparison.png")
    
    # 打印排名
    print("\n" + "="*60)
    print("Model Performance Ranking (sorted by Test Accuracy)")
    print("="*60)
    print(f"{'Model':<15} {'Train Acc':>12} {'Test Acc':>12} {'Gap':>10}")
    print("-"*60)
    
    results_list = [(k, results[k]['train_acc'], results[k]['test_acc']) for k in model_keys]
    results_sorted = sorted(results_list, key=lambda x: x[2], reverse=True)
    
    for rank, (model, train, test) in enumerate(results_sorted, 1):
        gap = train - test
        print(f"#{rank} {model:<12} {train:>10.2f}% {test:>10.2f}% {gap:>9.2f}%")
    
    plt.show()


if __name__ == "__main__":
    # 检查是否已有结果文件
    if os.path.exists('ntufi_results.json'):
        print("Found existing results file. Loading...")
        with open('ntufi_results.json', 'r') as f:
            results = json.load(f)
        
        # 询问是否重新运行
        choice = input("Do you want to re-run experiments? (y/n): ").strip().lower()
        if choice == 'y':
            results = run_all_experiments()
    else:
        results = run_all_experiments()
    
    plot_comparison(results)
