# 数据目录路径及预处理配置
IMG_PATH = '../common/dataset/'  # 原始图像存储根目录（需确保存在子目录）
FASHION_LABELS_PATH = '../common/fashion-labels.csv'  # 原始图像存储根目录（需确保存在子目录）
IMG_H = 64  # 原始图像高度（注意：实际训练时会被Resize为64x64）
IMG_W = 64  # 原始图像宽度（需检查与数据预处理的一致性）

# 随机性相关配置
SEED = 42  # 全局随机种子（确保实验可复现性）
TRAIN_RATIO = 0.75  # 训练集划分比例（75%训练，25%验证）
TEST_RATIO = 1 - TRAIN_RATIO  # 验证集比例（自动计算，无需修改）
SHUFFLE_BUFFER_SIZE = 100  # 数据混洗缓冲区大小（影响数据加载顺序随机性）

# 训练超参数
LEARNING_RATE = 1e-3  # 初始学习率（AdamW优化器使用）
TRAIN_BATCH_SIZE = 32  # 训练批次大小（GPU显存不足时可调小）
TEST_BATCH_SIZE = 32  # 测试批次大小（建议与训练批次一致）
VAL_BATCH_SIZE = 32  # 验证批次大小（建议与训练批次一致）
EPOCHS = 20  # 总训练轮次（需平衡过拟合与欠拟合）

# 模型接口相关配置
PACKAGE_NAME = 'image_classification'  # 模块包名
CLASSIFIER_MODEL_NAME = 'classifier.pt'  # 自编码器模型保存路径（未实际使用）

# 定义一个字典，将数字标签映射为对应的中文名称
classification_names = {
    0: '上衣',
    1: '鞋',
    2: '包',
    3: '下身衣服',
    4: '手表'
}
