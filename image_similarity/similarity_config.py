# 数据目录路径及预处理配置
IMG_PATH = '../common/dataset/'
IMG_H = 64  # 输入图像高度
IMG_W = 64  # 输入图像宽度

# 随机性相关配置
SEED = 42
TRAIN_RATIO = 0.75      # 训练集划分比例
TEST_RATIO = 1 - TRAIN_RATIO

# 训练超参数
LEARNING_RATE = 1e-3        # 学习率
TRAIN_BATCH_SIZE = 32       # 训练集批大小
VAL_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
FULL_BATCH_SIZE = 32        # 全数据集批大小（为生成图片嵌入，写入向量数据库）
EPOCHS = 30                 # 总训练轮次数

# 模型接口相关配置
PACKAGE_NAME = 'image_similarity'    # 模块包名
ENCODER_MODEL_NAME = 'encoder.pt'         # 编码器模型参数保存文件
DECODER_MODEL_NAME = 'decoder.pt'         # 解码器模型参数保存文件
EMBEDDING_NAME = 'embeddings.npy'         # 特征向量嵌入文件（向量数据库）