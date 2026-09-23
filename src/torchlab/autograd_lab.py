"""手写两层 MLP 的前向与反向传播，并与 torch.autograd 对拍。"""

# autograd_lab.py
import numpy as np
import torch

def forward_mlp(a0: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray):
    """
    前向传播：两层MLP，tanh隐藏层，输出无激活
    a0: [n, d_in]
    返回：z1,a1,z2 （缓存，给反向用）
    """
    z1 = a0 @ W1 + b1
    a1 = np.tanh(z1)
    z2 = a1 @ W2 + b2
    return z1, a1, z2


def backward_mlp(a0, W1, b1, W2, b2, z1, a1, z2, y: np.ndarray):
    """
    反向传播，和之前手推公式完全一致，numpy版本
    返回 dW1, db1, dW2, db2
    """
    n = a0.shape[0]
    d_out = y.shape[1]

    delta2 = 2 * (z2 - y) / (n * d_out)
    dW2 = a1.T @ delta2
    db2 = delta2.sum(axis=0)

    da1 = delta2 @ W2.T
    delta1 = da1 * (1 - a1**2)
    dW1 = a0.T @ delta1
    db1 = delta1.sum(axis=0)

    return dW1, db1, dW2, db2


def check_gradients(n=8, d_in=5, h=7, d_out=3, seed=42):
    """
    造numpy随机数据，跑自己的numpy前向+反向；再用torch复算，对比梯度，返回最大误差
    全部float64，消除精度差异
    """
    np.random.seed(seed)
    # numpy 参数
    a0_np = np.random.randn(n, d_in)
    W1_np = np.random.randn(d_in, h)
    b1_np = np.random.randn(h)
    W2_np = np.random.randn(h, d_out)
    b2_np = np.random.randn(d_out)
    y_np = np.random.randn(n, d_out)

    # 1. 自己numpy前向反向
    z1_np, a1_np, z2_np = forward_mlp(a0_np, W1_np, b1_np, W2_np, b2_np)
    dW1_np, db1_np, dW2_np, db2_np = backward_mlp(a0_np, W1_np, b1_np, W2_np, b2_np, z1_np, a1_np, z2_np, y_np)

    # 2. 转到torch，float64！
    a0_t = torch.tensor(a0_np, dtype=torch.float64)
    W1_t = torch.tensor(W1_np, dtype=torch.float64, requires_grad=True)
    b1_t = torch.tensor(b1_np, dtype=torch.float64, requires_grad=True)
    W2_t = torch.tensor(W2_np, dtype=torch.float64, requires_grad=True)
    b2_t = torch.tensor(b2_np, dtype=torch.float64, requires_grad=True)
    y_t = torch.tensor(y_np, dtype=torch.float64)

    # torch前向
    z1_t = a0_t @ W1_t + b1_t
    a1_t = torch.tanh(z1_t)
    z2_t = a1_t @ W2_t + b2_t
    loss = torch.sum((z2_t - y_t)**2) / (n * d_out)
    loss.backward()

    # 取出torch梯度
    dW1_t = W1_t.grad.numpy()
    db1_t = b1_t.grad.numpy()
    dW2_t = W2_t.grad.numpy()
    db2_t = b2_t.grad.numpy()

    # 逐项求绝对误差，找全局最大误差
    err_W1 = np.max(np.abs(dW1_np - dW1_t))
    err_b1 = np.max(np.abs(db1_np - db1_t))
    err_W2 = np.max(np.abs(dW2_np - dW2_t))
    err_b2 = np.max(np.abs(db2_np - db2_t))
    max_err = max(err_W1, err_b1, err_W2, err_b2)
    return max_err, err_W1, err_b1, err_W2, err_b2


if __name__ == "__main__":
    max_err, err_W1, err_b1, err_W2, err_b2 = check_gradients()
    print(f"全局最大误差 = {max_err:.2e}")
    print(f"W1 err:{err_W1:.2e}, b1 err:{err_b1:.2e}, W2 err:{err_W2:.2e}, b2 err:{err_b2:.2e}")
    assert max_err < 1e-6, "梯度对拍失败！"
    print("✅ numpy版本梯度和torch对拍通过！")
