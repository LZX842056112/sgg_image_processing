# 定义模块的公开接口，仅暴露ImageDataset类
__all__ = ['create_dataset']

import os  # 导入os库，用于处理文件和目录路径
import torch
from PIL import Image  # 导入PIL库中的Image模块，用于图像处理
from torch.utils.data import Dataset, random_split  # 从PyTorch的工具库中导入Dataset类，用于自定义数据集
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


# 定义一个名为ImageDataset的类，继承自PyTorch的Dataset类
class ImageDataset(Dataset):
    # 初始化
    def __init__(self, main_dir, transform=None):
        self.main_dir = main_dir  # 将主目录路径保存为类的属性
        self.transform = transform  # 将图像预处理操作保存为类的属性
        self.imgs = sorted_alphanum(os.listdir(main_dir))  # 获取所有图像文件名并按字母数字顺序排序

    # 定义__len__方法，返回数据集中图像的数量
    def __len__(self):
        return len(self.imgs)  # 返回主目录下图像文件的数量

    # 定义__getitem__方法，根据索引idx获取图像
    def __getitem__(self, idx):
        # 1. 根据索引号，找到文件名，拼接图片完整路径
        img_path = os.path.join(self.main_dir, self.imgs[idx])
        # 2. 使用PIL库打开图像并将其转换为RGB格式
        image = Image.open(img_path).convert('RGB')
        # 3. 如果定义了图像预处理操作
        if self.transform is not None:
            # 对图像进行预处理，并将其转换为张量
            tensor_image = self.transform(image)
        else:
            # 若无变换，抛出异常提示必须提供预处理
            raise ValueError("transform 参数不能为 None！")
        # 4. 向输入图像添加随机噪声
        # 生成与 tensor_image 形状相同的随机噪声，乘以噪声因子 noise_factor
        noisy_image = tensor_image + torch.randn_like(tensor_image) * NOISE_RATIO
        # 将图像像素值裁剪到 [0, 1] 范围内，避免超出有效范围
        noisy_image = torch.clamp(noisy_image, 0., 1.)
        # 返回预处理后的图像张量（将噪声图片作为输入，将原图作为目标）
        return noisy_image, tensor_image


# 自定义创建数据集函数
def create_dataset():
    # 定义图像预处理流程
    transform = T.Compose([
        T.Resize((IMG_H, IMG_W)),
        T.ToTensor()
    ])
    # 实例化完整数据集（输入和目标均为同一图像，自监督学习）
    dataset = ImageDataset(IMG_PATH, transform)
    # print(dataset.imgs)
    # 划分训练集和测试集（验证集）
    train_dataset, test_dataset = random_split(dataset, [TRAIN_RATIO, TEST_RATIO])

    return train_dataset, test_dataset


# 测试
if __name__ == '__main__':
    train_dataset, test_dataset = create_dataset()
    print(len(train_dataset))
    print(len(test_dataset))
