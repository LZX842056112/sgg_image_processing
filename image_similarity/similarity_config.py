# 数据目录路径及预处理配置
IMG_PATH = '../common/dataset/'  # 原始图像存储根目录（需确保存在子目录）
IMG_H = 64  # 原始图像高度（注意：实际训练时会被Resize为64x64）
IMG_W = 64  # 原始图像宽度（需检查与数据预处理的一致性）

# 随机性相关配置
SEED = 42  # 全局随机种子（确保实验可复现性）
TRAIN_RATIO = 0.75  # 训练集划分比例（75%训练，25%验证）
TEST_RATIO = 1 - TRAIN_RATIO  # 验证集比例（自动计算，无需修改）

# 训练超参数
LEARNING_RATE = 1e-3  # 初始学习率（AdamW优化器使用）
TRAIN_BATCH_SIZE = 32  # 训练批次大小（GPU显存不足时可调小
VAL_BATCH_SIZE = 32  # 验证批次大小（建议与训练批次一致）
TEST_BATCH_SIZE = 32  # 测试批次大小（建议与训练批次一致）
FULL_BATCH_SIZE = 32  # 全量数据生成嵌入时的批次大小（为生成图片嵌入，写入向量数据库）
EPOCHS = 30  # 总训练轮次（需平衡过拟合与欠拟合）

# 模型接口相关配置
PACKAGE_NAME = 'image_similarity'  # 模块包名
ENCODER_MODEL_NAME = 'encoder.pt'  # 编码器权重保存路径（需写权限）
DECODER_MODEL_NAME = 'decoder.pt'  # 解码器权重保存路径（需写权限）
EMBEDDING_NAME = 'embeddings.npy'  # 特征嵌入存储路径（.npy格式）
