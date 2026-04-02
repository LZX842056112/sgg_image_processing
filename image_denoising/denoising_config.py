# 数据目录路径及预处理配置
IMG_PATH = '../common/dataset/'
IMG_H = 68  # 输入图像高度
IMG_W = 68  # 输入图像宽度

# 随机性相关配置
SEED = 42
TRAIN_RATIO = 0.75      # 训练集划分比例
TEST_RATIO = 1 - TRAIN_RATIO
NOISE_RATIO = 0.5       # 噪声因子

# 训练超参数
LEARNING_RATE = 1e-3        # 学习率
TRAIN_BATCH_SIZE = 32       # 训练集批大小
VAL_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
EPOCHS = 30                 # 总训练轮次数

# 模型接口相关配置
PACKAGE_NAME = 'image_denoising'    # 模块包名
DENOISER_MODEL_NAME = 'denoiser.pt'         # 去噪器模型参数保存文件