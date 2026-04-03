import torch
from torch import nn, optim
from torch.utils.data import DataLoader  # 数据加载器
from tqdm import tqdm  # 进度条工具

from common.utils import seed_everything
from common.engine import train_epoch  # 训练引擎
from classification_engine import test_epoch  # 测试引擎

from classification_config import *
from classification_data import create_dataset  # 创建数据集
from classification_model import Classifier  # 模型

# 训练主流程
if __name__ == '__main__':
    # 基本准备工作
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 调用工具函数设置全局随机种子（确保可复现性）
    seed_everything(SEED)

    # 1. 创建数据集
    train_dataset, val_dataset = create_dataset()
    print(len(train_dataset), len(val_dataset))
    print("=============1. 数据集创建完成=============")

    # 2. 训练数据加载器（打乱顺序，丢弃最后不完整的批次）
    train_loader = DataLoader(train_dataset, TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
    # 验证数据加载器（不打乱，完整加载）
    val_loader = DataLoader(val_dataset, VAL_BATCH_SIZE, shuffle=False)
    print("============2. 数据加载器创建完成============")

    # 3. 定义模型、损失函数和优化器
    classifier = Classifier()
    classifier.to(device)

    # 指定损失函数为交叉熵损失函数，适用于多分类任务
    loss = nn.CrossEntropyLoss()
    # 指定优化器为Adam优化器，优化对象是模型的所有参数，学习率为0.001
    optimizer = optim.AdamW(classifier.parameters(), lr=LEARNING_RATE)
    print("=============3. 模型创建完成=============")

    # 4. 训练模型
    # 定义最小验证误差，用于判断是否保存模型
    min_val_loss = float('inf')

    # 循环迭代轮次
    for epoch in tqdm(range(EPOCHS)):
        # 执行一轮训练
        train_loss = train_epoch(classifier, train_loader, loss, optimizer, device)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss:.6f}")

        # 执行一次验证过程
        val_loss, val_acc = test_epoch(classifier, val_loader, loss, device)
        # 打印当前epoch的训练损失
        print(f"\n----------> Epochs = {epoch + 1}, Training Loss : {train_loss} <----------")

        # 模型保存逻辑：当验证损失创新低时保存模型
        if val_loss < min_val_loss:
            print("验证损失减小，保存模型...")
            # 保存编码器和解码器状态字典
            torch.save(classifier.state_dict(), CLASSIFIER_MODEL_NAME)
            min_val_loss = val_loss
        else:
            print("验证损失没有减小，不保存模型。")
        # 打印验证损失
        print(f"Epochs = {epoch + 1}, Validation Loss : {val_loss}")

    print("=============4. 模型训练完成=============")
    print("最终验证损失为：", min_val_loss)
