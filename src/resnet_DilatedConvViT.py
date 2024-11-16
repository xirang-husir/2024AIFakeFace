# -*- coding: utf-8 -*-
# @Time    : 2024/11/7
# @Author  : Husir
# @File    : resnet_dilated_vit_se.py
# @Software: PyCharm
# @Description : 构建 ResNet50 + 空洞卷积 + Vision Transformer + SE 通道注意力模块

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np


# 定义 Squeeze-and-Excitation (SE) 模块
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        """
        SE 通道注意力模块
        :param channel: 输入通道数
        :param reduction: 缩减率，用于控制瓶颈层的通道数
        """
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze：全局平均池化
        y = self.avg_pool(x).view(b, c)
        # Excitation：全连接层
        y = self.fc(y).view(b, c, 1, 1)
        # Scale：重新校准通道特征
        return x * y.expand_as(x)


# 定义包含 SEBlock 和空洞卷积的 Bottleneck 模块
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, dilation=1, downsample=None, reduction=16):
        super(Bottleneck, self).__init__()
        width = planes

        # 1x1 卷积，降低维度
        self.conv1 = nn.Conv2d(in_planes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        # 3x3 卷积，可能包含空洞
        self.conv2 = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(width)
        # 1x1 卷积，恢复维度
        self.conv3 = nn.Conv2d(
            width, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        # 添加 SEBlock
        self.se = SEBlock(planes * self.expansion, reduction)

    def forward(self, x):
        identity = x  # 残差连接

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # 应用 SEBlock
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # 残差连接
        out = self.relu(out)

        return out


# 定义 ViT 模块
class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super(ViTBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [序列长度, 批量大小, 嵌入维度]
        x2 = self.norm1(x)
        attn_output, _ = self.attn(x2, x2, x2)
        x = x + attn_output  # 残差连接

        x2 = self.norm2(x)
        x = x + self.mlp(x2)  # 残差连接

        return x


# 定义 ResNet50 + 空洞卷积 + ViT + SE 模型
class ResNet50_Dilated_ViT(nn.Module):
    def __init__(self, num_classes=2, replace_stride_with_dilation=None,
                 vit_dim=512, vit_depth=4, vit_heads=8, vit_mlp_ratio=4.0):
        super(ResNet50_Dilated_ViT, self).__init__()
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, True, True]
        self.in_planes = 64  # 初始通道数

        # 初始卷积层和最大池化层
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 定义可能包含空洞卷积的 ResNet 层
        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=1,  # 将步幅设为1
                                       dilate=replace_stride_with_dilation[1])

        # ViT 模块嵌入
        self.vit_embed_dim = vit_dim
        self.vit_patch_size = 1  # 特征图的每个像素作为一个 patch
        self.vit_num_patches = (28 * 28)  # 特征图尺寸为 [B, 1024, 28, 28]
        # 位置嵌入
        self.vit_pos_embed = nn.Parameter(torch.zeros(1, self.vit_num_patches, vit_dim))
        self.vit_dropout = nn.Dropout(p=0.1)
        # 线性映射，将特征图通道数映射到 vit_dim
        self.vit_proj = nn.Linear(1024, vit_dim)

        # ViT Transformer 编码器
        self.vit_blocks = nn.ModuleList([
            ViTBlock(dim=vit_dim, num_heads=vit_heads, mlp_ratio=vit_mlp_ratio)
            for _ in range(vit_depth)
        ])
        self.vit_norm = nn.LayerNorm(vit_dim)

        # 更新 self.in_planes，以匹配 ViT 输出通道数
        self.in_planes = self.vit_embed_dim

        # ResNet layer4，可能包含空洞卷积
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=1,  # 将步幅设为1
                                       dilate=replace_stride_with_dilation[2])

        # 全局平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        # 初始化权重
        self._initialize_weights()

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        """
        构建包含可选空洞卷积的 ResNet 层
        """
        downsample = None
        previous_dilation = 1
        dilation = previous_dilation

        if dilate:
            dilation *= stride
            stride = 1
        else:
            dilation = 1

        if stride != 1 or self.in_planes != planes * block.expansion:
            # 当输入和输出的尺寸或通道数不匹配时，应用下采样
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, dilation, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        # 初始化位置嵌入
        nn.init.trunc_normal_(self.vit_pos_embed, std=0.02)

        # 初始化其他权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 卷积层使用 Kaiming 初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                # 初始化 BatchNorm 和 LayerNorm
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 输入： [B, 3, 224, 224]
        x = self.conv1(x)  # [B, 64, 112, 112]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # [B, 64, 56, 56]

        x = self.layer1(x)  # [B, 256, 56, 56]
        x = self.layer2(x)  # [B, 512, 28, 28]
        x = self.layer3(x)  # [B, 1024, 28, 28]（由于空洞卷积）

        # ViT 模块
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, C] -> [B, N, 1024]
        x = self.vit_proj(x)  # [B, N, vit_dim]
        x = x + self.vit_pos_embed  # 添加位置嵌入
        x = self.vit_dropout(x)
        x = x.transpose(0, 1)  # [N, B, vit_dim]

        for blk in self.vit_blocks:
            x = blk(x)

        x = self.vit_norm(x)
        x = x.transpose(0, 1)  # [B, N, vit_dim]
        x = x.transpose(1, 2).reshape(B, self.vit_embed_dim, H, W)  # 恢复特征图形状

        # ResNet layer4
        x = self.layer4(x)  # [B, 2048, H', W']（H' 和 W' 取决于空洞率）

        x = self.avgpool(x)  # [B, 2048, 1, 1]
        x = torch.flatten(x, 1)  # [B, 2048]
        x = self.fc(x)  # [B, num_classes]

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

    # 创建模型实例
    model = ResNet50_Dilated_ViT(
        num_classes=2,
        replace_stride_with_dilation=[False, True, True],
        vit_dim=512,
        vit_depth=4,
        vit_heads=8
    )
    model.to(device)
    model.eval()  # 切换到评估模式

    # 计算模型参数总量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总量: {total_params}")

    # 创建一个样例输入图像（随机生成）
    sample_image = Image.fromarray(np.uint8(np.random.rand(130, 130, 3) * 255)).convert('RGB')

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
