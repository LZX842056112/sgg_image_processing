__all__ = ['train_epoch', 'test_epoch', 'create_embeddings', 'compute_similar_images']

import torch


# 定义一个轮次epoch训练的函数
def train_epoch(encoder, decoder, train_loader, loss, optimizer, device):
    encoder.train()
    decoder.train()

    totoal_loss = 0  # 记录累计训练损失

    for input, target in train_loader:
        input, target = input.to(device), target.to(device)
        # 1. 前向传播
        encoded_feature = encoder(input)
        output = decoder(encoded_feature)
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
def test_epoch(encoder, decoder, test_loader, loss, device):
    encoder.eval()
    decoder.eval()

    total_loss = 0

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
    encoder.eval()
    # 初始化张量，用来保存和返回嵌入矩阵
    embeddings = torch.empty(0)

    with torch.no_grad():
        for input, target in full_loader:
            input = input.to(device)  # 只需要用输入图像数据
            # 编码器前向传播
            output = encoder(input).cpu()
            # 将当前批次图像的特征（N=32, 256, 2, 2）拼接到嵌入张量中
            embeddings = torch.cat((embeddings, output), dim=0)

    # 将嵌入张量（N, 256, 2, 2）转换为二维嵌入矩阵（N, 1024）
    embeddings = embeddings.reshape(embeddings.shape[0], -1).numpy()

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
