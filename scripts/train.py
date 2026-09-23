# scripts/train.py
import argparse
import json
import logging
import os

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from torchlab.model import ResCNN, SmallCNN

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_loaders(batch_size, limit=None):
    """MNIST 数据加载器。limit 不为 None 时只取前 limit 条训练样本（跑通 pipeline 用）。"""
    trans = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root="./data/raw", train=True, download=True, transform=trans)
    val_ds = datasets.MNIST(root="./data/raw", train=False, download=True, transform=trans)

    if limit is not None:
        train_ds = Subset(train_ds, indices=list(range(limit)))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, criterion, device):
    """model.train() → 按 batch 跑五步，每 100 batch 打日志，返回 epoch 平均 loss。"""
    model.train()
    total_loss = 0.0
    n_batch = len(loader)
    for idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (idx + 1) % 100 == 0:
            logger.info(f"  Batch {idx+1}/{n_batch}, batch loss={loss.item():.4f}")
    return total_loss / n_batch


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """model.eval() → 返回 (val_loss, val_acc)。"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return total_loss / len(loader), correct / total


def main(args):
    torch.manual_seed(42)
    torch.set_num_threads(4)
    device = torch.device("cpu")

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    train_loader, val_loader = get_loaders(batch_size=args.batch_size, limit=args.limit)

    model = ResCNN() if args.model == "res" else SmallCNN()
    model = model.to(device)
    logger.info(
        f"模型: {type(model).__name__} | 参数量: {sum(p.numel() for p in model.parameters()):,}"
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    early_stop_counter = 0
    patience = 2

    # 关键：不同模型的产物分开存，防止 ResCNN 覆盖 SmallCNN 的结果
    suffix = args.model
    ckpt_path = f"checkpoints/best_{suffix}.pt"
    history_path = f"reports/history_{suffix}.json"

    for ep in range(1, args.epochs + 1):
        logger.info(f"==== Epoch {ep}/{args.epochs} ====")
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)
        logger.info(f"Epoch {ep} | train_loss={tr_loss:.4f}, val_loss={vl_loss:.4f}, val_acc={vl_acc:.4f}")

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"New best val loss, saved {ckpt_path}")
        else:
            early_stop_counter += 1
            logger.info(f"Val loss not improved. counter={early_stop_counter}/{patience}")
            if early_stop_counter >= patience:
                logger.info("Early stop triggered")
                break

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Saved history to {history_path} | final val_acc={history['val_acc'][-1]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="只取前 N 条训练样本，快速校验 pipeline")
    parser.add_argument("--epochs", type=int, required=True, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=64, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--model", choices=["cnn", "res"], default="cnn", help="模型类型")
    main(parser.parse_args())
