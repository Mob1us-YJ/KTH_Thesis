import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report


class NPZDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, indices: np.ndarray):
        self.X = X
        self.y = y
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        i = self.indices[idx]
        x = self.X[i]  # [T, F]
        x = torch.from_numpy(x).permute(1, 0)  # -> [F, T] for Conv1d
        y = int(self.y[i])
        return x.float(), y


def build_encoder(num_classes: int = 6, in_channels: int = 16) -> nn.Module:
    # 简单时间卷积分类器，输入 [B, 16, 256]
    class Encoder(nn.Module):
        def __init__(self, num_classes: int, in_ch: int):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(in_ch, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Conv1d(64, 128, kernel_size=5, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Linear(128, num_classes)

        def forward(self, x):
            feat = self.encoder(x).squeeze(-1)  # [B, 128]
            logits = self.head(feat)
            return logits, feat

    return Encoder(num_classes, in_channels)


def stratified_split(indices: np.ndarray, labels: np.ndarray, val_ratio: float, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    for cls in np.unique(labels):
        cls_mask = labels == cls
        cls_indices = indices[cls_mask]
        cls_perm = rng.permutation(cls_indices)
        n_val = max(1, int(len(cls_perm) * val_ratio)) if len(cls_perm) > 1 else 1
        val_idx.extend(cls_perm[:n_val].tolist())
        train_idx.extend(cls_perm[n_val:].tolist())
    return np.array(train_idx, dtype=np.int64), np.array(val_idx, dtype=np.int64)


def train_epoch(model, loader, device, criterion, optimizer):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
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
    all_y, all_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss_sum += loss.item() * xb.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += xb.size(0)
            all_y.append(yb.cpu().numpy())
            all_pred.append(pred.cpu().numpy())
    y_true = np.concatenate(all_y) if all_y else np.array([])
    y_pred = np.concatenate(all_pred) if all_pred else np.array([])
    return loss_sum / total, correct / total, y_true, y_pred


def main():
    p = argparse.ArgumentParser(description="Target-only WiSe4Car training split by car")
    p.add_argument("--data_dir", default="Transfer_Learning/Data/Wise4Car", help="目录包含 wise4car_unified_*.npy")
    p.add_argument("--car_id", type=int, required=True, help="选择的车型 ID，如 1,2,...；为 -1 时使用 car_id==-1 的样本")
    p.add_argument("--val_ratio", type=float, default=0.2, help="验证集占比")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="Transfer_Learning/result/wise4car_car_split", help="模型保存目录")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    feats = np.load(data_dir / "wise4car_unified_features.npy")  # [N,256,16]
    labels = np.load(data_dir / "wise4car_unified_labels.npy")   # [N]
    car_ids = np.load(data_dir / "wise4car_unified_car_ids.npy") # [N]
    label_names = np.load(data_dir / "wise4car_unified_label_names.npy")

    # 过滤指定 car_id
    mask = car_ids == args.car_id
    if mask.sum() == 0:
        raise ValueError(f"未找到 car_id={args.car_id} 的样本")
    feats = feats[mask]
    labels = labels[mask]

    # 构建划分（分层按类别）
    all_indices = np.arange(len(labels))
    train_idx, val_idx = stratified_split(all_indices, labels, args.val_ratio, seed=args.seed)
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("划分失败：训练或验证样本数量为 0")

    train_ds = NPZDataset(feats, labels, train_idx)
    val_ds = NPZDataset(feats, labels, val_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    model = build_encoder(num_classes=6, in_channels=feats.shape[2]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.out_dir, exist_ok=True)
    best_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, device, criterion, optimizer)
        va_loss, va_acc, y_true, y_pred = eval_epoch(model, val_loader, device, criterion)
        print(f"Epoch {epoch}: train_loss={tr_loss:.4f} acc={tr_acc:.4f} | val_loss={va_loss:.4f} acc={va_acc:.4f}")

        torch.save(model.state_dict(), os.path.join(args.out_dir, f"car{args.car_id}_epoch{epoch}.pth"))
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = model.state_dict()
            torch.save(best_state, os.path.join(args.out_dir, f"car{args.car_id}_best.pth"))

    # 用最佳模型输出混淆矩阵和分类报告
    if best_state is not None:
        model.load_state_dict(best_state)
    _, _, y_true, y_pred = eval_epoch(model, val_loader, device, criterion)

    labels_present = sorted(np.unique(np.concatenate([y_true, y_pred]))) if len(y_true) else []
    class_names = [str(label_names[i]) for i in labels_present]
    cm = confusion_matrix(y_true, y_pred, labels=labels_present)
    report = classification_report(y_true, y_pred, labels=labels_present, target_names=class_names, digits=4, zero_division=0)

    txt_path = Path(args.out_dir) / f"confusion_car{args.car_id}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Model: cnn\n")
        f.write("Confusion matrix (rows=true, cols=pred)\n")
        f.write(np.array2string(cm, separator=", "))
        f.write("\n\nClassification report\n")
        f.write(report)

    print(f"Done. Best val acc: {best_acc:.4f}. Saved confusion to {txt_path}")


if __name__ == "__main__":
    main()
