import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class NPZDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, indices: np.ndarray):
        self.X = features
        self.y = labels
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        i = self.indices[idx]
        x = self.X[i]  # [T, F]
        x = torch.from_numpy(x).permute(1, 0)  # -> [F, T] for Conv1d(C=F, L=T)
        y = int(self.y[i])
        return x.float(), y


def build_encoder(num_classes: int = 6, in_channels: int = 16, dropout: float = 0.3, width: int = 64) -> nn.Module:
    # 时间卷积编码器，适配 16 个频率通道，输入 [B, 16, T]
    class Encoder(nn.Module):
        def __init__(self, num_classes: int, in_ch: int):
            super().__init__()
            c1, c2, c3, c4 = width, width * 2, width * 4, width * 8
            self.encoder = nn.Sequential(
                nn.Conv1d(in_ch, c1, kernel_size=5, padding=2),
                nn.BatchNorm1d(c1),
                nn.ReLU(inplace=True),
                nn.Conv1d(c1, c2, kernel_size=5, padding=2),
                nn.BatchNorm1d(c2),
                nn.ReLU(inplace=True),
                nn.Conv1d(c2, c3, kernel_size=5, padding=2),
                nn.BatchNorm1d(c3),
                nn.ReLU(inplace=True),
                nn.Conv1d(c3, c4, kernel_size=5, padding=2),
                nn.BatchNorm1d(c4),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(1),
                nn.Dropout(dropout),
            )
            self.head = nn.Linear(c4, num_classes)

        def forward(self, x):
            feat = self.encoder(x).squeeze(-1)  # [B, 256]
            logits = self.head(feat)
            return logits, feat

    return Encoder(num_classes, in_channels)


def train_epoch(model, loader, device, criterion, optimizer, mixup_alpha: float = 0.0):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    mixup_enabled = mixup_alpha > 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()

        if mixup_enabled:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            perm = torch.randperm(xb.size(0), device=device)
            xb_mix = lam * xb + (1 - lam) * xb[perm]
            yb_perm = yb[perm]
            logits, _ = model(xb_mix)
            loss = lam * criterion(logits, yb) + (1 - lam) * criterion(logits, yb_perm)
        else:
            logits, _ = model(xb)
            loss = criterion(logits, yb)

        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * xb.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += xb.size(0)

    return loss_sum / total, correct / total


def eval_epoch(model, loader, device, criterion):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss_sum += loss.item() * xb.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += xb.size(0)
    return loss_sum / total, correct / total


def parse_args():
    p = argparse.ArgumentParser(description="Train encoder on preprocessed XRF55 features")
    p.add_argument("--data_dir", default="preprocessed", help="目录包含 XRF55_*.npy")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", default="result/preproc_run", help="模型保存目录")
    p.add_argument("--in_channels", type=int, default=16, help="与特征维度一致 (pooled_subcarriers)")
    p.add_argument("--use_sampler", action="store_true", help="使用 WeightedRandomSampler 缓解类别不平衡")
    p.add_argument("--early_stop_patience", type=int, default=12)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mixup_alpha", type=float, default=0.0, help=">0 启用 mixup 数据增强")
    p.add_argument("--scheduler", choices=["step", "cosine"], default="cosine")
    p.add_argument("--width", type=int, default=64, help="通道宽度基数，模型容量控制")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # 读取预处理数据
    X = np.load(os.path.join(args.data_dir, "XRF55_features.npy"))  # [N, 256, 16]
    y = np.load(os.path.join(args.data_dir, "XRF55_labels.npy"))    # [N]
    train_idx = np.load(os.path.join(args.data_dir, "XRF55_train_indices.npy"))
    val_idx = np.load(os.path.join(args.data_dir, "XRF55_val_indices.npy"))

    # 构建数据集/加载器
    train_ds = NPZDataset(X, y, train_idx)
    val_ds = NPZDataset(X, y, val_idx)

    if args.use_sampler:
        # 类别权重：频次的倒数
        unique, counts = np.unique(y[train_idx], return_counts=True)
        freq = dict(zip(unique.tolist(), counts.tolist()))
        class_weights = np.zeros(int(max(unique)) + 1, dtype=np.float32)
        for k, v in freq.items():
            class_weights[int(k)] = 1.0 / (v + 1e-6)
        sample_weights = class_weights[y[train_idx].astype(int)]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=args.num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    device = torch.device(args.device)
    # 类别权重 + label smoothing
    unique, counts = np.unique(y[train_idx], return_counts=True)
    freq = dict(zip(unique.tolist(), counts.tolist()))
    class_weights = torch.ones(int(max(unique)) + 1, dtype=torch.float32)
    for k, v in freq.items():
        class_weights[int(k)] = 1.0 / (v + 1e-6)
    class_weights = class_weights.to(device)

    model = build_encoder(num_classes=6, in_channels=args.in_channels, dropout=args.dropout, width=args.width).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    best_state = None
    patience = args.early_stop_patience
    wait = 0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, device, criterion, optimizer, mixup_alpha=args.mixup_alpha)
        va_loss, va_acc = eval_epoch(model, val_loader, device, criterion)
        print(f"Epoch {epoch}: train_loss={tr_loss:.4f} acc={tr_acc:.4f} | val_loss={va_loss:.4f} acc={va_acc:.4f}")

        torch.save(model.state_dict(), os.path.join(args.out_dir, f"encoder_epoch{epoch}.pth"))
        scheduler.step()

        if va_acc > best_acc:
            best_acc = va_acc
            wait = 0
            best_state = model.state_dict()
            torch.save(best_state, os.path.join(args.out_dir, "encoder_best.pth"))
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch} (best acc={best_acc:.4f})")
                break

    print(f"Best val acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
