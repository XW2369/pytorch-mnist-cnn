import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallCNN(nn.Module):
    """Conv(1→16,k=3,p=1)→BN→ReLU→MaxPool→Conv(16→32,k=3,p=1)→BN→ReLU→MaxPool→
       Flatten→Linear(1568→128)→ReLU→Dropout(0.5)→Linear(128→10)
       1568 = 32*7*7（你自己算的，第 3 条手推）
       ✅硬规则遵守：Conv→BN→ReLU；Dropout只放在全连接之前
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(1568,128),
            nn.ReLU(),
            nn.Dropout(0.5),  # dropout仅在FC前，ResBlock不放dropout
            nn.Linear(128,10)
        )

    def forward(self,x):
        return self.net(x)


class ResBlock(nn.Module):
    """两个 Conv(3×3,p=1)+BN 的残差分支，out = F(x) + x。
    难点：in_channels != out_channels 时，shortcut 用 1×1 Conv + BN 升维，
         让 x 和 F(x) 形状一致才能相加——这个分支的通道数要动脑子。
    ✅硬规则：Conv→BN→ReLU，ResBlock内没有Dropout
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # shortcut分支：通道/尺寸不匹配时，用1x1卷积+BN映射
        if in_channels != out_channels or stride !=1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        # F(x) 主分支
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 残差相加
        shortcut_x = self.shortcut(x)
        out = out + shortcut_x
        out = F.relu(out)
        return out


class ResCNN(nn.Module):
    """conv → ResBlock×2 → pool → fc。自己定合理结构，参数量必须 > SmallCNN。
    MNIST输入 1通道28×28
    """
    def __init__(self):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.res_blocks = nn.Sequential(
            ResBlock(32, 48, stride=1),
            ResBlock(48, 64, stride=2), # 这里stride=2，同时扩通道，下采样
        )
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        # 64通道 * 7*7
        self.fc1 = nn.Linear(64*7*7, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.res_blocks(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def count_parameters(model: nn.Module) -> int:
    """可训练参数总数。提示：requires_grad 为 True 的 p.numel() 求和。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 测试代码（验证shape+参数量）
if __name__ == "__main__":
    sm_cnn = SmallCNN()
    res_cnn = ResCNN()
    print("SmallCNN 可训练参数：", count_parameters(sm_cnn))
    print("ResCNN 可训练参数：", count_parameters(res_cnn))
    # 随机模拟MNIST batch：batch_size=8, channel=1, H=28,W=28
    dummy = torch.randn(8,1,28,28)
    print("SmallCNN out shape:", sm_cnn(dummy).shape)
    print("ResCNN out shape:", res_cnn(dummy).shape)
