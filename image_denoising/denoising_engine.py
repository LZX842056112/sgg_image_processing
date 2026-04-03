# 定义模块的公开接口
__all__ = ['train_epoch', 'test_epoch']

# 导入PyTorch核心库和神经网络模块
import torch


# 定义一个轮次epoch训练的函数
def train_epoch(denoiser, train_loader, loss, optimizer, device):
    """
        执行一个完整的训练迭代

        参数:
        - encoder: 卷积编码器（如ConvEncoder）
        - decoder: 卷积解码器（如ConvDecoder）
        - train_loader: 训练数据加载器，提供批次化的（输入图像, 目标图像）
        - loss: 损失函数（如MSE）
        - optimizer: 优化器（如Adam）
        - device: 计算设备（"cuda" 或 "cpu"）

        返回值:
        - 当前epoch的平均训练损失（标量值）
    """
    # 设置为训练模式（启用Dropout/BatchNorm等训练专用层），当前场景下无用
    # encoder.train()
    # decoder.train()
    denoiser.train()

    totoal_loss = 0  # 累计损失
    # 遍历训练数据加载器中的所有批次
    for input, target in train_loader:
        # 将数据移动到指定设备（GPU/CPU）
        input, target = input.to(device), target.to(device)
        # 1. 前向传播
        output = denoiser(input)
        # 2. 计算重建损失（预测图像与目标图像的差异）
        loss_value = loss(output, target)
        # 3. 反向传播：计算梯度
        loss_value.backward()
        # 4. 优化器更新模型参数
        optimizer.step()
        # 5. 清空优化器中之前的梯度
        optimizer.zero_grad()

        # 累加当前轮次损失
        totoal_loss += loss_value.item()
    # 返回本轮次训练平均损失
    return totoal_loss / len(train_loader)


# 定义一次测试（验证）的函数，返回评价指标（损失值）
def test_epoch(denoiser, test_loader, loss, device):
    """
        执行验证步骤（不更新参数）

        参数与train_epoch类似，但不含优化器参数

        返回值:
        - 验证集的平均损失（标量值）
    """
    # 设置为评估模式（禁用Dropout/BatchNorm等训练专用层）
    # encoder.eval()
    # decoder.eval()
    denoiser.eval()
    total_loss = 0
    # 禁用梯度计算以节省内存和计算资源
    with torch.no_grad():
        for input, target in test_loader:
            input, target = input.to(device), target.to(device)
            # 前向传播（推理预测）
            output = denoiser(input)
            # 计算损失
            loss_value = loss(output, target)
            # 累计损失
            total_loss += loss_value.item()
    # 计算测试（验证）平均损失
    test_loss = total_loss / len(test_loader)
    return test_loss
