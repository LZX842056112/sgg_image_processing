__all__ = ['create_dataset']

import os
import torch
from PIL import Image
from torch.utils.data import Dataset, random_split  # 数据集和随机划分
import torchvision.transforms as T  # 图像转换预处理

from similarity_config import *  # 引入自定义配置

from common.utils import sorted_alphanum


# 自定义图像数据集类
class ImageDataset(Dataset):
    # 初始化
    def __init__(self, main_dir, transform=None):
        self.main_dir = main_dir
        self.transform = transform
        self.imgs = sorted_alphanum(os.listdir(main_dir))  # 获取主目录下的所有图片文件名，保存为列表

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

        return tensor_image, tensor_image  # 返回 （输入图像，原图像）


# 自定义创建数据集函数
def create_dataset():
    # 定义转换操作
    transform = T.Compose([
        T.Resize((IMG_H, IMG_W)),
        T.ToTensor()
    ])
    # 定义数据集对象实例
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
