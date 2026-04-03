__all__ = ['Classifier']

import torch
import torch.nn as nn  # 导入 PyTorch 的神经网络模块，用于构建神经网络


class Classifier(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.model = nn.Sequential(
            # 定义第一个卷积层，输入通道数为3，输出通道数为8，卷积核大小为3x3，步幅为1，填充为1
            # 使用填充为1的卷积操作，输出特征图的尺寸与输入相同（Same convolutions）
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 定义最大池化层，池化核大小为2x2，步幅为2
            nn.MaxPool2d(2, 2),
            # 定义第二个卷积层，输入通道数为8，输出通道数为16，卷积核大小为3x3，步幅为1，填充为1
            # 上一层的输出通道数为8，因此这一层的输入通道数为8
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 将特征图展平为一维向量，保留batch_size维度，其余维度展平
            nn.Flatten(),
            # 定义全连接层，输入大小为16*16*16，输出大小为num_classes（分类数），全连接层的输入的尺寸计算在forward函数中解释
            nn.Linear(16 * 16 * 16, num_classes)
        )

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    # 创建一个随机输入张量，维度为1x3x32x32，用于测试模型
    x = torch.randn(10, 3, 64, 64)
    # 定义模型
    model = Classifier()
    # 前向传播
    y = model(x)
    print(y.shape)
