"""手搓训练循环：forward → loss → backward → step → zero_grad。"""
import logging
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class TinyMLP(nn.Module):
    """两层：Linear(20→32) → ReLU → Linear(32→2)。结构自己写 __init__ 和 forward。"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)


def manual_train_step(model, X_batch, y_batch, optimizer, criterion):
    """一个 batch 的训练。五步一行不能少，顺序不能换：
    1. optimizer.zero_grad()   ← 不写这条，梯度累加，loss 震荡不降（静默错误，不报错）
    2. out = model(X_batch)
    3. loss = criterion(out, y_batch)
    4. loss.backward()
    5. optimizer.step()
    返回 loss.item()
    """
    # 五步严格顺序
    optimizer.zero_grad()
    out = model(X_batch)
    loss = criterion(out, y_batch)
    loss.backward()
    optimizer.step()
    return loss.item()


def train_manual(epochs=30, batch_size=64, lr=1e-3, seed=42):
    """完整训练：
    - make_classification(n_samples=2000, n_features=20, random_state=seed)
    - train_test_split 分 80/20，张量统一 float64（和块 1A 同理，float32 精度太松）
    - 每 epoch：按 batch 循环调 manual_train_step → 记录 train_loss 均值 → eval 模式算 val acc
    - logging.info 每个 epoch 打一行：epoch / train_loss / val_acc
    - 返回 history（dict: train_loss 列表 + val_acc 列表）
    """
    # 固定随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 生成数据集 2000样本，20特征
    X, y = make_classification(n_samples=2000, n_features=20, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)

    # 转张量，统一 float64
    X_train = torch.tensor(X_train, dtype=torch.float64)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float64)
    y_val = torch.tensor(y_val, dtype=torch.long)

    model = TinyMLP()
    model.double()  # 模型参数转float64，消除dtype mismatch

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_acc": []}
    n_train = len(X_train)

    for epoch in range(1, epochs+1):
        model.train()
        batch_loss_list = []
        # 遍历batch
        perm = torch.randperm(n_train)
        for start in range(0, n_train, batch_size):
            idx = perm[start: start + batch_size]
            Xb = X_train[idx]
            yb = y_train[idx]
            loss_val = manual_train_step(model, Xb, yb, optimizer, criterion)
            batch_loss_list.append(loss_val)

        avg_train_loss = np.mean(batch_loss_list)
        history["train_loss"].append(avg_train_loss)

        # =====评估段，严格遵守 model.eval() + torch.no_grad()=====
        model.eval()
        with torch.no_grad():
            logits = model(X_val)
            pred = torch.argmax(logits, dim=1)
            acc = (pred == y_val).float().mean().item()
        model.train() # 切回训练模式

        history["val_acc"].append(acc)
        logger.info(f"epoch={epoch:2d} | train_loss={avg_train_loss:.4f} | val_acc={acc:.4f}")

    return history


if __name__ == "__main__":
    hist = train_manual(epochs=30, batch_size=64, lr=1e-3, seed=42)
    final_acc = hist["val_acc"][-1]
    logging.info(f"\n✅ 最后一轮验证集准确率: {final_acc:.4f}")
    assert final_acc > 0.85, f"末轮acc {final_acc:.4f} 未达到>0.85要求！"
