__all__ = ['create_dataset']

import os
import torch
from PIL import Image  # 图像处理库
from torch.utils.data import Dataset, random_split  # 数据集和随机划分
import torchvision.transforms as T  # 图像转换预处理

from similarity_config import *  # 引入自定义配置

from common.utils import sorted_alphanum


# 自定义图像数据集类
class ImageDataset(Dataset):
    """
    从图像文件夹创建PyTorch数据集，返回图像的张量表示

    参数:
    - main_dir : 图片存储路径（字符串）
    - transform (可选) : 图像预处理变换（如torchvision.transforms）
    """

    # 初始化
    def __init__(self, main_dir, transform=None):
        self.main_dir = main_dir  # 图像主目录
        self.transform = transform  # 图像预处理函数
        # 获取所有图像文件名并按字母数字顺序排序
        self.imgs = sorted_alphanum(os.listdir(main_dir))

    # 实现__len__方法
    def __len__(self):
        return len(self.imgs)

    # 实现__getitem__方法
    def __getitem__(self, idx):
        # 1. 根据索引号，找到文件名，拼接图片完整路径
        img_path = os.path.join(self.main_dir, self.imgs[idx])
        # 2. 加载图像，转换为RGB图像
        image = Image.open(img_path).convert('RGB')
        # 3. 做转换，得到张量
        if self.transform is not None:
            tensor_image = self.transform(image)
        else:
            raise ValueError("transform 参数不能为 None！")
        # 返回两次相同张量（输入和目标相同，用于自编码器训练）
        return tensor_image, tensor_image


# 自定义创建数据集函数
def create_dataset():
    # 定义图像预处理流程
    transform = T.Compose([
        T.Resize((IMG_H, IMG_W)),  # 统一缩放到64x64分辨率
        T.ToTensor()  # 转换为PyTorch张量（范围[0,1]）
    ])
    # 实例化完整数据集（输入和目标均为同一图像，自监督学习）
    dataset = ImageDataset(IMG_PATH, transform)
    # 划分训练集和测试集（验证集）
    train_dataset, test_dataset = random_split(dataset, [TRAIN_RATIO, TEST_RATIO])

    return dataset, train_dataset, test_dataset


# 测试
if __name__ == '__main__':
    dataset, train_dataset, test_dataset = create_dataset()
    print(len(dataset))
    print(len(train_dataset))
    print(len(test_dataset))
