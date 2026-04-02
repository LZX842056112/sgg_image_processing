__all__ = ['create_dataset']

import os
from PIL import Image
import pandas as pd
import torchvision.transforms as T
from torch.utils.data import Dataset, random_split
from common.utils import sorted_alphanum
from classification_config import *


# 自定义数据集类
class ImageLabelDataset(Dataset):
    # 初始化
    def __init__(self, main_dir, label_dir, transform=None):
        self.main_dir = main_dir
        self.transform = transform
        self.imgs = sorted_alphanum(os.listdir(main_dir))
        # 读取分类标签
        labels = pd.read_csv(label_dir)
        # 将分类标签保存为字典
        self.labels_dict = dict(zip(labels['id'], labels['target']))

    # 获取长度
    def __len__(self):
        return len(self.imgs)

    # 按照索引号获取元素
    def __getitem__(self, idx):
        # 1. 获取索引号idx对应图片文件路径
        img_path = os.path.join(self.main_dir, self.imgs[idx])
        # 2. 加载图像数据
        image = Image.open(img_path).convert('RGB')
        # 3. 对图像进行预处理转换
        if self.transform is not None:
            tensor_image = self.transform(image)
        else:
            raise ValueError("transform 参数不能为 None！")
        # 4. 在标签字典中查找对应分类标签
        img_label = self.labels_dict[idx]

        return tensor_image, img_label


# 创建数据集函数
def create_dataset():
    # 定义图像预处理转换
    transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor()
    ])
    dataset = ImageLabelDataset(IMG_PATH, FASHION_LABELS_PATH, transform)
    # 切分数据集
    train_dataset, test_dataset = random_split(dataset, [TRAIN_RATIO, TEST_RATIO])

    return train_dataset, test_dataset


if __name__ == '__main__':
    train_dataset, test_dataset = create_dataset()
    print(len(train_dataset))
    print(len(test_dataset))
