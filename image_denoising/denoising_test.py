import torch
from torch.utils.data import DataLoader
from denoising_config import *
from denoising_data import create_dataset
from denoising_model import ConvDenoiser
from denoising_engine import test_epoch

import matplotlib.pyplot as plt

# 提取一批测试数据，推理测试并对比画图
def test(model, test_loader, device):
    model.to(device)
    model.eval()
    # 1. 从测试集中获取一个批次的数据
    data_iter = iter(test_loader)
    noisy_images, images = next(data_iter)
    print("输入和目标图像形状：", noisy_images.shape, images.shape)

    # 2. 前向传播，推理测试
    with torch.no_grad():
        noisy_images = noisy_images.to(device)
        outputs = model(noisy_images)
        print("输出去噪图像形状：", outputs.shape)

    # 3. 转换图像数据，准备画图
    noisy_images = noisy_images.permute(0, 2, 3, 1).cpu().numpy()
    outputs = outputs.permute(0, 2, 3, 1).detach().cpu().numpy()
    original_images = images.permute(0, 2, 3, 1).cpu().numpy()

    # 4. 画图显示
    fig, axes = plt.subplots(3, 10, figsize=(25, 4), sharex=True, sharey=True)
    # 遍历每个子图，绘制对应图像
    for imgs, ax_row in zip([noisy_images, outputs, original_images], axes):
        for img, ax in zip(imgs, ax_row):
            ax.imshow(img)
            ax.axis('off')
    plt.show()

# 测试主流程
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 创建数据集并划分
    train_dataset, test_dataset = create_dataset()
    print("=============1. 数据集创建完成=============")

    # 2. 定义数据加载器
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE)
    print("============2. 数据加载器创建完成============")

    # 3. 定义模型，并从文件加载模型参数
    loaded_denoiser = ConvDenoiser()
    # 加载参数字典
    model_state_dict = torch.load(DENOISER_MODEL_NAME, map_location=device)
    loaded_denoiser.load_state_dict(model_state_dict)
    print("==============3. 模型加载完成 ==============")

    # 4. 测试
    print("==============4. 测试结果如下 ==============")
    test(loaded_denoiser, test_loader, device)

    test_loss = test_epoch( loaded_denoiser, test_loader, loss = torch.nn.MSELoss(), device=device)
    print("测试集平均误差：", test_loss)