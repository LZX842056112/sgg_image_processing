__all__ = ['test_epoch']

import torch


# 定义一次测试（验证）的函数，返回评价指标（损失值、准确率）
def test_epoch(model, test_loader, loss, device):
    model.eval()
    total_loss = 0  # 累加测试损失
    correct_num = 0  # 累加预测正确数量
    test_num = 0  # 测试集总样本数

    with torch.no_grad():
        for input, target in test_loader:
            input, target = input.to(device), target.to(device)
            # 前向传播（推理预测）
            output = model(input)
            # 计算损失
            loss_value = loss(output, target)
            # 累计损失
            total_loss += loss_value.item() * input.shape[0]  # 按样本数量求加权和
            test_num += input.shape[0]  # 累加测试集样本数
            # 统计本批次预测准确个数
            pred = output.argmax(dim=1)
            correct_num += pred.eq(target).sum().item()

    # 计算测试（验证）平均损失和准确率
    test_loss = total_loss / test_num
    accuracy = correct_num / test_num
    return test_loss, accuracy
