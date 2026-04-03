__all__ = ['train_epoch', 'test_epoch', 'create_embeddings', 'compute_similar_images']

import torch


# 定义一个轮次epoch训练的函数
def train_epoch(encoder, decoder, train_loader, loss, optimizer, device):
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
    encoder.train()
    decoder.train()

    totoal_loss = 0  # 记录累计训练损失
    # 遍历训练数据加载器中的所有批次
    for input, target in train_loader:
        input, target = input.to(device), target.to(device)
        # 1. 前向传播：编码器生成潜在表示
        encoded_feature = encoder(input)
        # 前向传播：解码器重建图像
        output = decoder(encoded_feature)
        # 2. 计算重建损失（预测图像与目标图像的差异）
        loss_value = loss(output, target)
        # 3. 反向传播：计算梯度
        loss_value.backward()
        # 4. 优化器更新模型参数
        optimizer.step()
        # 5. 梯度清零
        optimizer.zero_grad()

        # 累加当前轮次损失
        totoal_loss += loss_value.item()
    # 返回本轮次训练平均损失
    return totoal_loss / len(train_loader)


# 定义一次测试（验证）的函数，返回评价指标（损失值）
def test_epoch(encoder, decoder, test_loader, loss, device):
    """
    执行验证步骤（不更新参数）

    参数与train_step类似，但不含优化器参数

    返回值:
    - 验证集的平均损失（标量值）
    """
    encoder.eval()
    decoder.eval()

    total_loss = 0
    # 禁用梯度计算以节省内存和计算资源
    with torch.no_grad():
        for input, target in test_loader:
            input, target = input.to(device), target.to(device)
            # 前向传播（推理预测）
            encoded_feature = encoder(input)
            output = decoder(encoded_feature)
            # 计算损失
            loss_value = loss(output, target)
            # 累计损失
            total_loss += loss_value.item()
    # 计算测试（验证）平均损失
    test_loss = total_loss / len(test_loader)
    return test_loss


# 对全量数据集生成图片嵌入表达（编码器推理），返回ndarray
def create_embeddings(encoder, full_loader, device):
    """
    使用编码器为整个数据集生成嵌入表示

    参数:
    - encoder: 训练好的编码器
    - full_loader: 完整数据集的数据加载器
    - device: 计算设备

    返回值:
    - 嵌入张量，形状为 (num_samples + 1, c, h, w)
    """
    encoder.eval()
    # 初始化随机嵌入（注：此处可能应为空张量，初始随机值可能引入噪声）
    # embedding = torch.randn(embedding_dim)  # 初始随机张量
    embeddings = torch.empty(0)

    with torch.no_grad():
        for input, target in full_loader:
            # 仅使用输入图像（自编码器结构）
            input = input.to(device)
            # 生成当前批次的嵌入，将结果移回CPU
            output = encoder(input).cpu()
            # 将当前批次图像的特征（N=32, 256, 2, 2）拼接到嵌入张量中（可能导致内存问题，建议预分配空间）
            embeddings = torch.cat((embeddings, output), dim=0)

    # 将高维嵌入展平为二维数组（样本数 x 特征维度）
    # 将嵌入张量（N, 256, 2, 2）转换为二维嵌入矩阵（N, 1024）
    embeddings = embeddings.reshape(embeddings.shape[0], -1).numpy()
    # 返回所有样本的嵌入集合
    return embeddings


from sklearn.neighbors import NearestNeighbors  # 获取“最近邻”


# 计算相似图片：输入一张图片的张量数据，返回数据库中最相似的K个图片索引列表
def compute_similar_images(encoder, image_tensor, num_images, embeddings, device):
    # 1. 将图像移动至设备
    image_tensor = image_tensor.to(device)

    # 2. 前向传播，得到输入图像的嵌入表达（ 形状 (256, 2, 2) 特征），转为 ndarray
    with torch.no_grad():
        image_embedding = encoder(image_tensor).cpu().numpy()

    # 3. 转为二维结构，（N, m），m 为向量维度
    image_vector = image_embedding.reshape((image_embedding.shape[0], -1))

    # 4. 定义一个 KNN 模型
    knn = NearestNeighbors(n_neighbors=num_images, metric='cosine')

    # 5. 训练模型（在嵌入矩阵上拟合）
    knn.fit(embeddings)

    # 6. 查询k近邻
    _, indices = knn.kneighbors(image_vector)

    return indices.tolist()
