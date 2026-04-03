# 定义模块的公开接口
__all__ = ['ConvDenoiser']

import torch
import torch.nn as nn


# 自定义神经网络模型（去噪自编码器）
class ConvDenoiser(nn.Module):
    # 初始化
    def __init__(self):
        super().__init__()
        # 编码器部分：三层卷积-池化
        self.encoder = nn.Sequential(
            ## 编码器层 ##
            # 卷积层 (输入通道数从1变为32), 3x3卷积核
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 池化层，用于将x-y维度减少一半；卷积核和步幅均为2
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 卷积层 (输入通道数从32变为16), 3x3卷积核
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 池化层，用于将x-y维度减少一半；卷积核和步幅均为2
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 卷积层 (输入通道数从16变为8), 3x3卷积核
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 池化层，用于将x-y维度减少一半；卷积核和步幅均为2
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 解码器部分：三层转置卷积+普通卷积
        self.decoder = nn.Sequential(
            # 转置卷积层，卷积核为2，步幅为2，将空间维度增加2倍
            # 卷积核大小为3，以得到7x7的图像输出
            nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2),
            nn.ReLU(),
            # 另外两个转置卷积层，卷积核为2
            nn.ConvTranspose2d(8, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            # 最后一个普通的卷积层，用于减少通道数
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            # 输出应应用sigmoid函数
            nn.Sigmoid()
        )

    # 前向传播
    def forward(self, x):
        # 编码
        x = self.encoder(x)
        # print("编码器输出形状: ", x.shape)
        # 解码
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    # 创建一个随机输入张量
    input = torch.randn(1, 3, 68, 68)
    # 创建一个卷积编码器模型
    denoiser = ConvDenoiser()
    # 编码输入张量
    output = denoiser(input)
    print(output.shape)
