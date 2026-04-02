import torch
import numpy as np
import matplotlib.pyplot as plt

from similarity_config import *
from similarity_data import create_dataset
from similarity_model import ConvEncoder  # 编码器模型
from similarity_engine import compute_similar_images  # 计算相似图像

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 创建数据集，从测试集中取一个数据
    dataset, train_dataset, test_dataset = create_dataset()
    image, _ = test_dataset[0]
    image = image.unsqueeze(0)  # 升维，得到 (1, 3, 64, 64)
    print(image.shape)

    # 2. 加载模型
    loaded_encoder = ConvEncoder()
    # 从文件加载参数字典
    state_dict = torch.load(ENCODER_MODEL_NAME, map_location=device)
    # 加载参数
    loaded_encoder.load_state_dict(state_dict)
    loaded_encoder.to(device)

    # 3. 加载图像嵌入矩阵
    embeddings = np.load(EMBEDDING_NAME)

    # 4. 计算得到相似图片索引列表
    num_similar = 5
    similar_image_indices = compute_similar_images(loaded_encoder, image, num_images=num_similar, embeddings=embeddings,
                                                   device=device)

    print(similar_image_indices)

    # 5. 画图
    fig, axes = plt.subplots(2, 5, figsize=(25, 4))
    # 输入图片
    image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    axes[0, 2].imshow(image)
    # 相似图片
    for i in range(num_similar):
        # 取当前图片的索引号
        index = similar_image_indices[0][i]
        # 从数据集中取图片
        img, _ = dataset[index]
        # 转换
        img = img.permute(1, 2, 0).numpy()
        # 画图
        axes[1, i].imshow(img)

    for ax in axes.flat:
        ax.axis('off')

    plt.show()
