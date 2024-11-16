# -*- coding: utf-8 -*-
# @Time    : 2024/11/9 19:44
# @Author  : Husir
# @File    : resnet_ASPP.py
# @Software: PyCharm
# @Description : 将空洞给卷积升级为ASPP模块，提升特征提取的多样性。

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np


# 定义ASPP模块
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.atrous_block1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                                       bias=False)
        self.atrous_block3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=3, dilation=3,
                                       bias=False)
        self.atrous_block5 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=5, dilation=5,
                                       bias=False)
        self.atrous_block7 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=7, dilation=7,
                                       bias=False)

        # 全局平均池化层
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        )

        # 1x1卷积，用于整合不同空洞率的特征
        self.conv1 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 不同空洞率的卷积操作
        out1 = self.atrous_block1(x)
        out3 = self.atrous_block3(x)
        out5 = self.atrous_block5(x)
        out7 = self.atrous_block7(x)
        out_global = self.global_avg_pool(x)
        out_global = F.interpolate(out_global, size=x.shape[2:], mode='bilinear', align_corners=False)

        # 将所有输出连接在一起
        out = torch.cat([out1, out3, out5, out7, out_global], dim=1)
        out = self.conv1(out)
        out = self.bn(out)
        return self.relu(out)

# 定义Squeeze-and-Excitation Block
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        """
        Squeeze-and-Excitation Block
        :param channel: 输入特征图的通道数
        :param reduction: 缩减率，用于控制中间层的通道数
        """
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Squeeze操作
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: 全局平均池化
        y = self.avg_pool(x).view(b, c)
        # Excitation: 两层全连接网络
        y = self.fc(y).view(b, c, 1, 1)
        # Scale: 重新标定
        return x * y.expand_as(x)

# 定义Bottleneck模块，集成SEBlock
class Bottleneck(nn.Module):
    expansion = 4  # Bottleneck模块的通道扩张倍数

    def __init__(self, in_planes, planes, stride=1, dilation=1, downsample=None, reduction=16):
        super(Bottleneck, self).__init__()
        width = planes  # 中间层的通道数
        # 第一层：1x1卷积，降低维度
        self.conv1 = nn.Conv2d(in_planes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        # 第二层：3x3卷积，进行特征提取，使用空洞卷积
        self.conv2 = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            stride=stride,
            padding=dilation,  # 根据空洞率调整填充
            dilation=dilation,  # 空洞率
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(width)
        # 第三层：1x1卷积，还原通道数
        self.conv3 = nn.Conv2d(
            width, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # 残差连接中的下采样
        self.stride = stride

        # 添加 SEBlock
        self.se = SEBlock(planes * self.expansion, reduction)

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
            # 如果输入和输出的尺寸或通道数不一致，需要在捷径分支进行下采样
            identity = self.downsample(x)

        out += identity  # 残差连接
        out = self.relu(out)

        # 应用 SEBlock
        out = self.se(out)

        return out

# 定义ResNet50 + 空洞卷积 + SE通道注意力机制的模型
class ResNet50_Dilated_ASPP(nn.Module):
    def __init__(self, num_classes=2, replace_stride_with_dilation=[False, True, True]):
        super(ResNet50_Dilated_ASPP, self).__init__()
        self.in_planes = 64  # 初始输入通道数

        # 初始卷积层
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 定义ResNet层级
        self.layer1 = self._make_layer(Bottleneck, 64, 3, stride=1)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2, dilate=replace_stride_with_dilation[2])

        # 添加 ASPP 模块
        self.aspp = ASPP(512 * Bottleneck.expansion, 256)

        # 输出层
        self.fc = nn.Linear(256, num_classes)

        # 初始化权重
        self._initialize_weights()

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = 1
        if dilate:
            dilation = stride
            stride = 1
        else:
            dilation = 1

        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, dilation, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 使用 ASPP 提取多尺度特征
        x = self.aspp(x)

        # 全局平均池化
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# 定义图像预处理
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 测试函数
def test_model():
    # 初始化设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建模型实例，二分类任务
    model = ResNet50_Dilated_ASPP(num_classes=2, replace_stride_with_dilation=[False, True, True])
    model.to(device)
    model.eval()  # 切换到评估模式

    # 计算模型参数总量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总量: {total_params}")

    # 创建一个样例输入图像
    sample_image = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255)).convert('RGB')

    # 预处理图像
    input_tensor = preprocess(sample_image).unsqueeze(0).to(device)  # [1, 3, 224, 224]

    # 前向传播
    with torch.no_grad():
        output = model(input_tensor)

    print(f"输入张量形状: {input_tensor.shape}")
    print(f"输出张量形状: {output.shape}")
    print(f"输出结果: {output}")

if __name__ == "__main__":
    test_model()