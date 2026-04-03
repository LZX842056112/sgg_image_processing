# 数据路径与预处理配置
IMG_PATH = '../common/dataset/'  # 原始图像存储根目录（需确保存在子目录）
IMG_H = 68  # 原始图像高度（注意：实际训练时会被Resize为68x68）
IMG_W = 68  # 原始图像宽度（需检查与数据预处理的一致性）

# 随机性与数据划分配置
SEED = 42  # 全局随机种子（确保实验可复现性）
TRAIN_RATIO = 0.75  # 训练集划分比例（75%训练，25%验证）
TEST_RATIO = 1 - TRAIN_RATIO  # 验证集比例（自动计算，无需修改）
SHUFFLE_BUFFER_SIZE = 100  # 数据混洗缓冲区大小（影响数据加载顺序随机性）
NOISE_RATIO = 0.5  # 设置噪声因子，用于向图像添加噪声

# 训练超参数
LEARNING_RATE = 1e-3  # 初始学习率（AdamW优化器使用）
TRAIN_BATCH_SIZE = 32  # 训练批次大小（GPU显存不足时可调小）
VAL_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32  # 验证/测试批次大小（建议与训练批次一致）
EPOCHS = 30  # 总训练轮次（需平衡过拟合与欠拟合）

# 模型接口相关配置
PACKAGE_NAME = 'image_denoising'  # 模块包名
DENOISER_MODEL_NAME = 'denoiser.pt'  # 编码器权重保存路径（需写权限）
