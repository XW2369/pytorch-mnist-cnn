"""Day4 单元测试：手写 MLP 反向传播 vs torch.autograd。

4 条用例：
1. W1 梯度对拍
2. W2 梯度对拍
3. b1 / b2 梯度对拍
4. 零梯度测试：y == z2（loss=0）时所有梯度必须为 0
"""

import numpy as np

from torchlab.autograd_lab import forward_mlp, backward_mlp, check_gradients


def test_W1_grad_match():
    errs = check_gradients()
    assert errs[1] < 1e-9, f"W1 err={errs[1]:.2e}"


def test_W2_grad_match():
    errs = check_gradients()
    assert errs[3] < 1e-9, f"W2 err={errs[3]:.2e}"


def test_bias_grad_match():
    errs = check_gradients()
    assert errs[2] < 1e-9, f"b1 err={errs[2]:.2e}"
    assert errs[4] < 1e-9, f"b2 err={errs[4]:.2e}"


def test_zero_grad_when_y_eq_z2():
    """y == z2 时 loss=0，四个梯度必须全为 0。

    专门抓「忘记除以 n*d_out」这类错误。
    """
    np.random.seed(0)
    n, d_in, h, d_out = 8, 5, 7, 3
    a0 = np.random.randn(n, d_in)
    W1 = np.random.randn(d_in, h)
    b1 = np.random.randn(h)
    W2 = np.random.randn(h, d_out)
    b2 = np.random.randn(d_out)

    z1, a1, z2 = forward_mlp(a0, W1, b1, W2, b2)
    y = z2.copy()

    dW1, db1, dW2, db2 = backward_mlp(a0, W1, b1, W2, b2, z1, a1, z2, y)

    eps = 1e-9
    assert np.all(np.abs(dW1) < eps)
    assert np.all(np.abs(db1) < eps)
    assert np.all(np.abs(dW2) < eps)
    assert np.all(np.abs(db2) < eps)
