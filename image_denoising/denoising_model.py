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
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 解码器部分：三层转置卷积+普通卷积
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
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
    # 定义输入数据
    input = torch.randn(1, 3, 68, 68)
    # 定义模型
    denoiser = ConvDenoiser()
    # 前向传播
    output = denoiser(input)
    print(output.shape)