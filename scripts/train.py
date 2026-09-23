# scripts/train.py
import argparse
import json
import logging
import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchlab.model import SmallCNN

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_loaders(batch_size, limit=None):
    """MNIST，transform 用 ToTensor（得到 [0,1] float32）。
    limit 不为 None 时用 Subset 只取前 limit 条（跑通 pipeline 用）。
    """
    trans = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=trans)
    val_ds = datasets.MNIST(root="./data", train=False, download=True, transform=trans)

    if limit is not None:
        train_ds = Subset(train_ds, indices=list(range(limit)))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, criterion, device):
    """model.train() → 按batch五步，每100 batch打日志，返回epoch平均loss"""
    model.train()
    total_loss = 0.0
    n_batch = len(loader)
    for idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        # 五步：前向、算loss、清零梯度、反向、参数更新
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (idx + 1) % 100 == 0:
            logger.info(f"  Batch {idx+1}/{n_batch}, batch loss={loss.item():.4f}")
    avg_loss = total_loss / n_batch
    return avg_loss


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """model.eval()，返回(val_loss, val_acc)"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item()
        _, pred = torch.max(logits, dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    avg_loss = total_loss / len(loader)
    acc = correct / total
    return avg_loss, acc


def main(args):
    torch.manual_seed(42)
    torch.set_num_threads(4)   # 限制CPU线程，防止占满卡死
    device = torch.device("cpu") # 严格只用CPU，无cuda分支

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    train_loader, val_loader = get_loaders(batch_size=64, limit=args.limit)
    model = SmallCNN().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    early_stop_counter = 0
    patience = 2

    for ep in range(args.epochs):
        logger.info(f"==== Epoch {ep+1}/{args.epochs} ====")
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)
        logger.info(f"Epoch {ep+1} | train_loss={tr_loss:.4f}, val_loss={vl_loss:.4f}, val_acc={vl_acc:.4f}")

        # 保存best checkpoint
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), "checkpoints/best.pt")
            logger.info("New best val loss, saved checkpoints/best.pt")
        else:
            early_stop_counter +=1
            logger.info(f"Val loss not improved. early_stop_counter={early_stop_counter}/{patience}")
            if early_stop_counter >= patience:
                logger.info("Early stop triggered")
                break

    # 输出history json
    with open("reports/history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logger.info("Saved history to reports/history.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="只取前N条训练样本，快速pipeline校验")
    parser.add_argument("--epochs", type=int, required=True, help="训练轮数")
    args = parser.parse_args()
    main(args)
