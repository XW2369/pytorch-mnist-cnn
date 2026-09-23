# scripts/plot_curve.py
"""从 history_*.json 画训练曲线：train_loss（左轴）+ val_acc（右轴）。"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无界面环境也能出图
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def plot(history_path: str) -> str:
    with open(history_path, encoding="utf-8") as f:
        hist = json.load(f)

    epochs = range(1, len(hist["train_loss"]) + 1)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, hist["train_loss"], "o-", color="tab:blue", label="训练损失 train_loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(epochs, hist["val_acc"], "s--", color="tab:red", label="验证准确率 val_acc")
    ax2.set_ylabel("Val Acc", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title("MNIST 训练曲线（SmallCNN, CPU 2 epoch）")
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
    fig.tight_layout()

    out = Path("reports/figures/training_curve.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"已保存: {out}")
    return str(out)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "reports/history_cnn.json"
    plot(path)
