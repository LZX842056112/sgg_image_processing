__all__ = ['train_epoch', 'test_epoch']

import torch


# 定义一个轮次epoch训练的函数
def train_epoch(model, train_loader, loss, optimizer, device):
    model.train()

    totoal_loss = 0  # 记录累计训练损失

    for input, target in train_loader:
        input, target = input.to(device), target.to(device)
        # 1. 前向传播
        output = model(input)
        # 2. 计算损失
        loss_value = loss(output, target)
        # 3. 反向传播
        loss_value.backward()
        # 4. 更新参数
        optimizer.step()
        # 5. 梯度清零
        optimizer.zero_grad()

        # 累加当前轮次损失
        totoal_loss += loss_value.item()
    # 返回本轮次训练平均损失
    return totoal_loss / len(train_loader)


# 定义一次测试（验证）的函数，返回评价指标（损失值）
def test_epoch(model, test_loader, loss, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for input, target in test_loader:
            input, target = input.to(device), target.to(device)
            # 前向传播（推理预测）
            output = model(input)
            # 计算损失
            loss_value = loss(output, target)
            # 累计损失
            total_loss += loss_value.item()
    # 计算测试（验证）平均损失
    test_loss = total_loss / len(test_loader)
    return test_loss
