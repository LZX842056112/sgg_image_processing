__all__ = ['create_dataset']

import os
import torch
from PIL import Image
from torch.utils.data import Dataset, random_split  # 数据集和随机划分
import torchvision.transforms as T  # 图像转换预处理

from denoising_config import *  # 引入自定义配置

# import re
#
# # 自定义函数：对图片文件名进行字母-数字排序
# def sorted_alphanum(img_names):
#     convert = lambda str: int(str) if str.isdigit() else str.lower()
#     alphanum_key = lambda img_name: [convert(str) for str in re.split(r'([0-9]+)', img_name)]
#     return sorted(img_names, key=alphanum_key)

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
        # 4. 基于原图加入噪声，构建输入数据
        noisy_image = tensor_image + torch.randn_like(tensor_image) * NOISE_RATIO
        # 将噪声图像数据裁剪到[0, 1]范围内
        noisy_image = torch.clamp(noisy_image, 0., 1.)

        return noisy_image, tensor_image  # 返回 （噪声图像，原图像）


# 自定义创建数据集函数
def create_dataset():
    # 定义转换操作
    transform = T.Compose([
        T.Resize((IMG_H, IMG_W)),
        T.ToTensor()
    ])
    # 定义数据集对象实例
    dataset = ImageDataset(IMG_PATH, transform)
    # print(dataset.imgs)
    # 划分训练集和测试集（验证集）
    train_dataset, test_dataset = random_split(dataset, [TRAIN_RATIO, VAL_RATIO])

    return train_dataset, test_dataset


# 测试
if __name__ == '__main__':
    train_dataset, test_dataset = create_dataset()
    print(len(train_dataset))
    print(len(test_dataset))
