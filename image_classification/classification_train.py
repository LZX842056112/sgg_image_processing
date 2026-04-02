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
    seed_everything(SEED)

    # 1. 创建数据集
    train_dataset, val_dataset = create_dataset()
    print(len(train_dataset), len(val_dataset))
    print("=============1. 数据集创建完成=============")

    # 2. 定义数据加载器
    train_loader = DataLoader(train_dataset, TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, VAL_BATCH_SIZE, shuffle=False)
    print("============2. 数据加载器创建完成============")

    # 3. 定义模型、损失函数和优化器
    classifier = Classifier()
    classifier.to(device)

    loss = nn.CrossEntropyLoss()
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
        print(f"Epoch {epoch + 1}/{EPOCHS}, Validation Loss: {val_loss:.6f}, Validation Acc: {val_acc:.6f}")

        # 模型保存逻辑
        if val_loss < min_val_loss:
            print("验证损失减小，保存模型...")
            torch.save(classifier.state_dict(), CLASSIFIER_MODEL_NAME)
            min_val_loss = val_loss
        else:
            print("验证损失没有减小，不保存模型。")

    print("=============4. 模型训练完成=============")
    print("最终验证损失为：", min_val_loss)
