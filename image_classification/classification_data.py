__all__ = ['create_dataset']

import os
from PIL import Image  # 图像处理库
import pandas as pd
import torchvision.transforms as T
from torch.utils.data import Dataset, random_split
from common.utils import sorted_alphanum
from classification_config import *


# 自定义数据集类
class ImageLabelDataset(Dataset):
    """
    从图像文件夹创建PyTorch数据集，返回图像的张量表示

    参数:
    - main_dir : 图片存储路径（字符串）
    - transform (可选) : 图像预处理变换（如torchvision.transforms）
    """

    # 初始化
    def __init__(self, main_dir, label_dir, transform=None):
        self.main_dir = main_dir  # 图像主目录
        self.transform = transform  # 获取所有图像文件名并按字母数字顺序排序
        self.imgs = sorted_alphanum(os.listdir(main_dir))  # 图像文件列表
        # 读取包含分类标签的 CSV 文件
        labels = pd.read_csv(label_dir)
        # 将数据类型转换为字典，提升查询效率
        self.labels_dict = dict(zip(labels['id'], labels['target']))

    # 返回数据集中图像的总数量
    def __len__(self):
        return len(self.imgs)

    # 加载并返回指定索引的图像张量（输入和目标相同，适用于自编码器）
    def __getitem__(self, idx):
        # 1. 获取索引号idx对应图片文件路径
        img_path = os.path.join(self.main_dir, self.imgs[idx])
        # 2. 打开图像并转换为RGB格式
        image = Image.open(img_path).convert('RGB')
        # 3. 对图像进行预处理转换
        if self.transform is not None:
            tensor_image = self.transform(image)
        else:
            raise ValueError("transform 参数不能为 None！")
        # 4. 在标签字典中查找对应分类标签
        img_label = self.labels_dict[idx]
        # 返回图像张量和标签
        return tensor_image, img_label


# 创建数据集函数
def create_dataset():
    # 定义图像预处理转换
    transform = T.Compose([
        T.Resize((64, 64)),  # 统一缩放到64x64分辨率
        T.ToTensor()  # 转换为PyTorch张量（范围[0,1]）
    ])
    # 实例化完整数据集（输入和目标均为同一图像，自监督学习）
    dataset = ImageLabelDataset(IMG_PATH, FASHION_LABELS_PATH, transform)
    # 随机划分数据集
    train_dataset, test_dataset = random_split(dataset, [TRAIN_RATIO, TEST_RATIO])

    return train_dataset, test_dataset


if __name__ == '__main__':
    train_dataset, test_dataset = create_dataset()
    print(len(train_dataset))
    print(len(test_dataset))
