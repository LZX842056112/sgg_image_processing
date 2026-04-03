import torch
from torch import nn, optim  # 引入神经网络和优化器
from torch.utils.data import DataLoader  # 数据加载器

from tqdm import tqdm  # 进度条工具

from common.utils import *
from denoising_config import *
from denoising_data import create_dataset
from denoising_model import ConvDenoiser
from denoising_engine import *

# 训练主流程
if __name__ == '__main__':
    # 检测GPU可用性并设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 调用工具函数设置全局随机种子（确保可复现性）
    seed_everything(SEED)

    # 1. 创建数据集并划分
    train_dataset, val_dataset = create_dataset()
    print(len(train_dataset), len(val_dataset))
    print("=============1. 数据集创建完成=============")

    # 2. 训练数据加载器（打乱顺序，丢弃最后不完整的批次）
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
    # 验证数据加载器（不打乱，完整加载）
    val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE)
    print("============2. 数据加载器创建完成============")

    # 3. 初始化自编码器用于去噪
    # 创建自编码器实例
    denoiser = ConvDenoiser()
    denoiser.to(device)
    # 均方误差损失函数
    loss = nn.MSELoss()
    # 指定优化器
    optimizer = optim.AdamW(denoiser.parameters(), lr=LEARNING_RATE)
    print("=============3. 模型创建完成=============")

    # 4. 训练模型
    # 初始化最佳损失值为极大值，用于判断是否保存模型
    min_val_loss = float('inf')
    # 使用tqdm进度条遍历预设的epoch数量
    for epoch in tqdm(range(EPOCHS)):
        # 调用一个轮次的训练
        train_loss = train_epoch(denoiser, train_loader, loss, optimizer, device)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss:.6f}")
        # 调用一次验证（测试）
        val_loss = test_epoch(denoiser, val_loader, loss, device)
        # 打印当前epoch的训练损失
        print(f"\n----------> Epochs = {epoch + 1}, Training Loss : {train_loss:.6f} <----------")

        # 判断如果验证损失创新低，就保存当前模型参数
        if val_loss < min_val_loss:
            print("验证损失减小，保存模型...")
            # 保存去噪自编码器状态字典
            torch.save(denoiser.state_dict(), DENOISER_MODEL_NAME)
            min_val_loss = val_loss
        else:
            print("验证损失没有减小，不保存模型。")
        # 打印验证损失
        print(f"Epochs = {epoch + 1}, Validation Loss : {val_loss:.6f}")
    print("=============4. 模型训练完成=============")
    print("最终验证损失为：", min_val_loss)
