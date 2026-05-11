import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset


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
        x = torch.from_numpy(x).permute(1, 0)  # -> [F, T]
        y = int(self.y[i])
        return x.float(), y


def build_encoder(num_classes: int = 6, in_channels: int = 16) -> nn.Module:
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
                nn.Conv1d(128, 256, kernel_size=5, padding=2),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Linear(256, num_classes)

        def forward(self, x):
            feat = self.encoder(x).squeeze(-1)  # [B, 256]
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
        n_val = max(1, int(round(len(cls_perm) * val_ratio))) if len(cls_perm) > 1 else 1
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
    acc = correct / total if total else 0.0
    loss_avg = loss_sum / total if total else 0.0
    return loss_avg, acc, y_true, y_pred


def group_metrics_by_car(y_true: np.ndarray, y_pred: np.ndarray, car_ids: np.ndarray) -> Dict[int, Dict[str, float]]:
    metrics: Dict[int, Dict[str, float]] = {}
    for cid in sorted(np.unique(car_ids).tolist()):
        mask = car_ids == cid
        yt = y_true[mask]
        yp = y_pred[mask]
        if len(yt) == 0:
            metrics[cid] = {"acc": float("nan"), "macro_f1": float("nan"), "support": 0}
            continue
        metrics[cid] = {
            "acc": float(accuracy_score(yt, yp)),
            "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
            "support": int(len(yt)),
        }
    return metrics


def main():
    p = argparse.ArgumentParser(description="Train single model on all cars, report per-car metrics")
    p.add_argument("--data_dir", default="Transfer_Learning/Data/Wise4Car", help="Directory with wise4car_unified_*.npy")
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="Transfer_Learning/result/wise4car_all_cars")
    p.add_argument("--save_metrics", action="store_true", help="Save metrics JSON to out_dir")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    feats = np.load(data_dir / "wise4car_unified_features.npy")  # [N,256,16]
    labels = np.load(data_dir / "wise4car_unified_labels.npy")   # [N]
    car_ids = np.load(data_dir / "wise4car_unified_car_ids.npy") # [N]

    all_indices = np.arange(len(labels))
    train_idx, val_idx = stratified_split(all_indices, labels, args.val_ratio, seed=args.seed)

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
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = model.state_dict()
            torch.save(best_state, os.path.join(args.out_dir, "best.pth"))
        torch.save(model.state_dict(), os.path.join(args.out_dir, f"epoch{epoch}.pth"))

    # Evaluate best model on val for group metrics
    if best_state is not None:
        model.load_state_dict(best_state)
    _, va_acc, y_true, y_pred = eval_epoch(model, val_loader, device, criterion)
    car_ids_val = car_ids[val_idx]
    group_metrics = group_metrics_by_car(y_true, y_pred, car_ids_val)

    print("\nPer-car metrics (val):")
    for cid, m in group_metrics.items():
        print(f" car {cid}: acc={m['acc']:.3f}, macro_f1={m['macro_f1']:.3f}, support={m['support']}")
    print(f"Val overall acc={va_acc:.3f}, samples={len(y_true)}")

    if args.save_metrics:
        out_path = Path(args.out_dir) / "metrics.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"overall_acc": va_acc, "group": group_metrics}, f, indent=2)
        print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
