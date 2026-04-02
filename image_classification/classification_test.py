import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from classification_config import *
from classification_data import create_dataset
from classification_model import Classifier
from classification_engine import test_epoch

# 定义测试函数
def test( model, test_loader, device ):
    model.eval()
    model.to(device)

    # 1. 获取测试集的一个批次数据
    data_iter = iter(test_loader)
    images, labels = next(data_iter)

    # 2. 前向传播，推理测试
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)
        print("输出预测标签形状为：", outputs.shape)

    # 3. 数据转换，为画图显示做准备
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    # 得到预测标签
    preds = outputs.argmax(dim=1).cpu().numpy()

    # 4. 画图和打印输出
    fig, axes = plt.subplots(1, 10, figsize=(25, 4), sharex=True, sharey=True)
    for i in range(10):
        # 画图
        axes[i].imshow(images[i])
        axes[i].axis('off')
        # 打印真实标签
        print(f"{i + 1}-label: {labels[i]}")
        # 打印预测标签
        class_pred = classification_names[preds[i]]
        print(f"{i + 1}-pred: {preds[i]}, 预测分类：{class_pred}")
        print()

    plt.show()

# 测试主流程
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 创建数据集
    train_dataset, test_dataset = create_dataset()
    print(len(train_dataset), len(test_dataset))
    print("=============1. 数据集创建完成=============")

    # 2. 定义测试集数据加载器
    test_loader = DataLoader(test_dataset, VAL_BATCH_SIZE, shuffle=False)
    print("============2. 数据加载器创建完成============")

    # 3. 定义模型并从文件中加载训练好的参数
    loaded_classifier = Classifier()
    # 加载参数字典
    model_state_dict = torch.load(CLASSIFIER_MODEL_NAME, map_location=device)
    loaded_classifier.load_state_dict(model_state_dict)
    print("==============3. 模型加载完成 ==============")

    # 4. 测试
    print("==============4. 测试结果如下 ==============")
    test( loaded_classifier, test_loader, device )

    test_loss, test_acc = test_epoch(loaded_classifier, test_loader, loss=nn.CrossEntropyLoss(), device=device)
    print(f"测试集平均误差：{test_loss:.6f}")
    print(f"测试分类准确率：{test_acc:.6f}")