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
    # 调用工具函数设置全局随机种子（确保可复现性）
    seed_everything(SEED)

    # 1. 创建数据集并划分
    dataset, train_dataset, val_dataset = create_dataset()
    print("=============1. 数据集创建完成=============")

    # 2. 定义数据加载器
    # 训练数据加载器（打乱顺序，丢弃最后不完整的批次）
    train_loader = DataLoader(dataset=train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
    # 验证数据加载器（不打乱，完整加载）
    val_loader = DataLoader(dataset=val_dataset, batch_size=VAL_BATCH_SIZE)
    # 全量数据加载器（用于生成嵌入）
    full_loader = DataLoader(dataset=dataset, batch_size=FULL_BATCH_SIZE, shuffle=False)

    print("============2. 数据加载器创建完成============")

    # 3. 定义模型、损失函数和优化器
    encoder = ConvEncoder()
    decoder = ConvDecoder()
    encoder.to(device)
    decoder.to(device)
    # 定义损失函数（均方误差损失）
    loss = nn.MSELoss()
    # 定义优化器（联合优化编码器和解码器参数）
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
        # 打印当前epoch的训练损失
        print(f"\n----------> Epochs = {epoch + 1}, Training Loss : {train_loss:.6f} <----------")

        # 模型保存逻辑：当验证损失创新低时保存模型
        if val_loss < min_val_loss:
            print("验证损失减小，保存模型...")
            # 保存编码器和解码器状态字典
            torch.save(encoder.state_dict(), ENCODER_MODEL_NAME)
            torch.save(decoder.state_dict(), DECODER_MODEL_NAME)
            min_val_loss = val_loss
        else:
            print("验证损失没有减小，不保存模型。")
            # 打印验证损失
            print(f"Epochs = {epoch + 1}, Validation Loss : {val_loss:.6f}")
    print("=============4. 模型训练完成=============")
    print("最终验证损失为：", min_val_loss)

    # 5. 生成图像嵌入矩阵
    # 5.1 从文件加载最优模型（编码器）
    encoder_state_dict = torch.load(ENCODER_MODEL_NAME)
    encoder.load_state_dict(encoder_state_dict)

    # 5.2 生成嵌入矩阵
    # 调用函数生成所有数据的嵌入表示
    embeddings = create_embeddings(encoder, full_loader, device)

    # 5.3 保存到文件（向量数据库，如Chroma）
    np.save(EMBEDDING_NAME, embeddings)

    print("嵌入矩阵形状：", embeddings.shape)

    print("=============5. 图像嵌入矩阵生成完成=============")
