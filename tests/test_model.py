# tests/test_model.py
import torch
import pytest
from torchlab.model import SmallCNN, ResBlock, ResCNN, count_parameters

def test_smallcnn_output_shape():
    """用例1：SmallCNN输入shape (2,1,28,28) → 输出(2,10)"""
    model = SmallCNN()
    x = torch.rand(2, 1, 28, 28)
    out = model(x)
    assert out.shape == torch.Size([2, 10])


def test_param_count_rescnn_larger():
    """用例2：ResCNN参数 > SmallCNN参数"""
    sm = SmallCNN()
    res = ResCNN()
    cnt_sm = count_parameters(sm)
    cnt_res = count_parameters(res)
    assert cnt_res > cnt_sm


def test_resblock_identity_mapping():
    """
    用例3：ResBlock恒等映射测试
    输入必须用 torch.rand（非负），不能用randn。
    原因：ResBlock主分支权重全部清零后，主分支输出为0；
    最后有ReLU，会把负值截断成0。如果用randn带负数输入，
    shortcut里的负值经过相加+ReLU会被截断，导致不等于原始输入，造成测试误判。
    torch.rand生成[0,1]非负张量，清零主分支后 out = 0 + shortcut(x)，ReLU不做截断，可以满足 out == x。
    """
    blk = ResBlock(in_channels=16, out_channels=16, stride=1)
    # 把残差主分支权重全部置0
    for m in [blk.conv1, blk.conv2, blk.bn1, blk.bn2]:
        m.weight.data.zero_()
        if m.bias is not None:
            m.bias.data.zero_()
    x = torch.rand(2,16,28,28) # ✅rand，不是randn
    out = blk(x)
    torch.testing.assert_close(out, x, rtol=0, atol=1e-5)


def test_bn_train_eval_behavior():
    """
    用例4：BN+Dropout train/eval行为差异
    train模式：BN用batch内统计量 + Dropout随机，两次前向结果不一样；
    eval模式：BN使用running均值方差，Dropout关闭，两次前向结果完全一致。
    """
    model = SmallCNN()
    x = torch.rand(2,1,28,28)

    model.train()
    o1_train = model(x)
    o2_train = model(x)
    assert not torch.allclose(o1_train, o2_train)

    model.eval()
    with torch.no_grad():
        o1_eval = model(x)
        o2_eval = model(x)
    torch.testing.assert_close(o1_eval, o2_eval)
