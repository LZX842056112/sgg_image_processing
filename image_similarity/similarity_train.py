import torch
from torch import nn, optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm  # 进度条工具

from common.utils import *
from similarity_config import *
from similarity_data import create_dataset
from similarity_model import ConvEncoder, ConvDecoder
from similarity_engine import *

# 训练主流程
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(SEED)

    # 1. 创建数据集并划分
    dataset, train_dataset, val_dataset = create_dataset()
    print("=============1. 数据集创建完成=============")

    # 2. 定义数据加载器
    train_loader = DataLoader(dataset=train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=VAL_BATCH_SIZE)
    full_loader = DataLoader(dataset=dataset, batch_size=FULL_BATCH_SIZE, shuffle=False)

    print("============2. 数据加载器创建完成============")

    # 3. 定义模型、损失函数和优化器
    encoder = ConvEncoder()
    decoder = ConvDecoder()
    encoder.to(device)
    decoder.to(device)
    # 损失函数：MSE
    loss = nn.MSELoss()
    # 优化器：AdamW
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.AdamW(params, lr=LEARNING_RATE)

    print("=============3. 模型创建完成=============")

    # 4. 训练模型
    # 定义最小验证误差，用于判断是否保存模型
    min_val_loss = float('inf')
    for epoch in tqdm(range(EPOCHS)):
        # 执行一轮训练
        train_loss = train_epoch(encoder, decoder, train_loader, loss, optimizer, device)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss:.6f}")
        # 执行一次验证过程
        val_loss = test_epoch(encoder, decoder, val_loader, loss, device)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Validation Loss: {val_loss:.6f}")

        # 模型保存逻辑
        if val_loss < min_val_loss:
            print("验证损失减小，保存模型...")
            torch.save(encoder.state_dict(), ENCODER_MODEL_NAME)
            torch.save(decoder.state_dict(), DECODER_MODEL_NAME)
            min_val_loss = val_loss
        else:
            print("验证损失没有减小，不保存模型。")

    print("=============4. 模型训练完成=============")
    print("最终验证损失为：", min_val_loss)

    # 5. 生成图像嵌入矩阵
    # 5.1 从文件加载最优模型（编码器）
    encoder_state_dict = torch.load(ENCODER_MODEL_NAME)
    encoder.load_state_dict(encoder_state_dict)

    # 5.2 生成嵌入矩阵
    embeddings = create_embeddings(encoder, full_loader, device)

    # 5.3 保存到文件（向量数据库，如Chroma）
    np.save(EMBEDDING_NAME, embeddings)

    print("嵌入矩阵形状：", embeddings.shape)

    print("=============5. 图像嵌入矩阵生成完成=============")
