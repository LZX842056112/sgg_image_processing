__all__ = ['ConvEncoder', 'ConvDecoder']

import torch
import torch.nn as nn


# 编码器
class ConvEncoder(nn.Module):
    # 初始化
    def __init__(self):
        super().__init__()
        # 5个卷积层
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        # 通用池化层
        self.pool = nn.MaxPool2d(2, stride=2)

    # 前向传播
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        # print("第一层卷积-池化后的形状：", x.shape)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        # print("第二层卷积-池化后的形状：", x.shape)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        # print("第三层卷积-池化后的形状：", x.shape)
        x = torch.relu(self.conv4(x))
        x = self.pool(x)
        # print("第四层卷积-池化后的形状：", x.shape)
        x = torch.relu(self.conv5(x))
        x = self.pool(x)
        return x


# 解码器
class ConvDecoder(nn.Module):
    # 初始化
    def __init__(self):
        super().__init__()
        # 5层转置卷积
        self.conv_t1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_t2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_t3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_t4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv_t5 = nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2)

    # 前向传播
    def forward(self, x):
        x = torch.relu(self.conv_t1(x))
        # print("第一层转置卷积后的形状：", x.shape)
        x = torch.relu(self.conv_t2(x))
        # print("第二层转置卷积后的形状：", x.shape)
        x = torch.relu(self.conv_t3(x))
        # print("第三层转置卷积后的形状：", x.shape)
        x = torch.relu(self.conv_t4(x))
        # print("第四层转置卷积后的形状：", x.shape)
        # 最后一层激活函数用 Sigmoid，将输出限制在(0, 1)范围内
        x = torch.sigmoid(self.conv_t5(x))
        return x


if __name__ == '__main__':
    # 定义输入数据
    input = torch.randn(10, 3, 64, 64)

    # 创建模型
    encoder = ConvEncoder()
    decoder = ConvDecoder()

    # 前向传播
    encoded_feature = encoder(input)
    output = decoder(encoded_feature)

    print()
    print("编码器输出特征形状：", encoded_feature.shape)
    print("解码器输出形状：", output.shape)
