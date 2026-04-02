# 数据目录路径及预处理配置
IMG_PATH = '../common/dataset/'
FASHION_LABELS_PATH = '../common/fashion-labels.csv'
IMG_H = 64
IMG_W = 64

# 随机性相关配置
SEED = 42
TRAIN_RATIO = 0.75
TEST_RATIO = 1 - TRAIN_RATIO

# 训练超参数
LEARNING_RATE = 1e-3
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
VAL_BATCH_SIZE = 32
EPOCHS = 20

# 模型接口相关配置
PACKAGE_NAME = 'image_classification'    # 模块包名
CLASSIFIER_MODEL_NAME = 'classifier.pt'         # 分裂期模型参数保存文件

# 定义一个字典，将分类标签映射为中文名称
classification_names = {
    0: '上衣',
    1: '鞋',
    2: '包',
    3: '下身衣服',
    4: '手表'
}