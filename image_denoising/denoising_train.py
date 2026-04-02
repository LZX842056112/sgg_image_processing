import torch
from torch import nn, optim     # 引入神经网络和优化器
from torch.utils.data import DataLoader     # 数据加载器

from tqdm import tqdm       # 进度条工具

from common.utils import *
from denoising_config import *
from denoising_data import create_dataset
from denoising_model import ConvDenoiser
from denoising_engine import *

# 训练主流程
if __name__ == '__main__':
    # 准备操作
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(SEED)   # 全局设置随机数种子，保证训练可复现性

    # 1. 创建数据集并划分
    train_dataset, val_dataset = create_dataset()
    print(len(train_dataset), len(val_dataset))
    print("=============1. 数据集创建完成=============")

    # 2. 定义数据加载器
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE)
    print("============2. 数据加载器创建完成============")

    # 3. 定义模型、损失函数和优化器
    denoiser = ConvDenoiser()
    denoiser.to(device)
    loss = nn.MSELoss()    # 均方误差损失函数
    optimizer = optim.AdamW(denoiser.parameters(), lr=LEARNING_RATE)
    print("=============3. 模型创建完成=============")

    # 4. 训练模型
    # 定义最小验证误差，用于判断是否保存模型
    min_val_loss = float('inf')
    for epoch in tqdm(range(EPOCHS)):
        # 调用一个轮次的训练
        train_loss = train_epoch(denoiser, train_loader, loss, optimizer, device)
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.6f}")
        # 调用一次验证（测试）
        val_loss = test_epoch(denoiser, val_loader, loss, device)
        print(f"Epoch {epoch+1}/{EPOCHS}, Validation Loss: {val_loss:.6f}")

        # 判断如果验证损失创新低，就保存当前模型参数
        if val_loss < min_val_loss:
            print("验证损失减小，保存模型...")
            torch.save( denoiser.state_dict(), DENOISER_MODEL_NAME )
            min_val_loss = val_loss
        else:
            print("验证损失没有减小，不保存模型。")

    print("=============4. 模型训练完成=============")
    print("最终验证损失为：", min_val_loss)

