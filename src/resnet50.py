
# -*- coding: utf-8 -*-
# @Time    : 2024/4/27
# @Author  : Husir
# @File    : standard_resnet50.py
# @Software: PyCharm
# @Description : 标准的 ResNet50 模型

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# 定义标准的 Bottleneck 模块
class Bottleneck(nn.Module):
    expansion = 4  # Bottleneck 模块的通道扩张倍数

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        width = planes  # 中间层的通道数
        # 第一层：1x1 卷积，降低维度
        self.conv1 = nn.Conv2d(in_planes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        # 第二层：3x3 卷积，进行特征提取
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        # 第三层：1x1 卷积，还原通道数
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x  # 保存输入用于残差连接

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # 残差连接
        out = self.relu(out)

        return out

# 定义标准的 ResNet-50 模型
class ResNet50(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet50, self).__init__()
        self.in_planes = 64  # 初始输入通道数

        # 初始卷积层
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        # 最大池化层，降低特征图尺寸
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 定义每个层级
        self.layer1 = self._make_layer(Bottleneck, 64, 3, stride=1)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

        # 全局平均池化层和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        # 初始化权重
        self._initialize_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        构建 ResNet 的层级
        :param block: 基本块类型（Bottleneck）
        :param planes: 基本块的输出通道数
        :param blocks: 基本块的数量
        :param stride: 步幅
        """
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion  # 更新输入通道数
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        前向传播
        :param x: 输入图像，形状 [B, 3, H, W]
        :return: 分类结果，形状 [B, num_classes]
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 测试函数
def test_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建模型实例，二分类任务
    model = ResNet50(num_classes=2)
    model.to(device)
    model.eval()

    # 计算模型参数总量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总量: {total_params}")

    # 创建一个样例输入图像
    sample_image = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255)).convert('RGB')

    # 预处理图像
    input_tensor = preprocess(sample_image).unsqueeze(0).to(device)

    # 前向传播
    with torch.no_grad():
        output = model(input_tensor)

    print(f"输入张量形状: {input_tensor.shape}")
    print(f"输出张量形状: {output.shape}")
    print(f"输出结果: {output}")

if __name__ == "__main__":
    test_model()
