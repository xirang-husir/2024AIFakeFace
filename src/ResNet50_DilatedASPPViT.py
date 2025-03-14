# -*- coding: utf-8 -*-
# @Time    : 2024/11/10 1:23
# @Author  : Husir
# @File    : ResNet50_DilatedASPPViT.py
# @Software: PyCharm
# @Description : 修改了ResNet50_DilatedViTASPP 中ViT模块的位置

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

# 定义 ASPP 模块
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.atrous_block1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.atrous_block3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=3, dilation=3, bias=False)
        self.atrous_block5 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=5, dilation=5, bias=False)
        self.atrous_block7 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=7, dilation=7, bias=False)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        )

        self.conv1 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out1 = self.atrous_block1(x)
        out3 = self.atrous_block3(x)
        out5 = self.atrous_block5(x)
        out7 = self.atrous_block7(x)
        out_global = self.global_avg_pool(x)
        out_global = F.interpolate(out_global, size=x.shape[2:], mode='bilinear', align_corners=False)

        out = torch.cat([out1, out3, out5, out7, out_global], dim=1)
        out = self.conv1(out)
        out = self.bn(out)
        return self.relu(out)

# 定义 Squeeze-and-Excitation (SE) 模块
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# 定义包含 SEBlock 和空洞卷积的 Bottleneck 模块
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, dilation=1, downsample=None, reduction=16):
        super(Bottleneck, self).__init__()
        width = planes
        self.conv1 = nn.Conv2d(in_planes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.se = SEBlock(planes * self.expansion, reduction)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

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
        x2 = self.norm1(x)
        attn_output, _ = self.attn(x2, x2, x2)
        x = x + attn_output
        x2 = self.norm2(x)
        return x + self.mlp(x2)

# 定义 ResNet50 + 空洞卷积 + ViT (at Tail) + ASPP + SE 模型
class ResNet50_Dilated_ASPP_ViT(nn.Module):
    def __init__(self, num_classes=2, replace_stride_with_dilation=None, vit_dim=512, vit_depth=4, vit_heads=8,
                 vit_mlp_ratio=4.0):
        super(ResNet50_Dilated_ASPP_ViT, self).__init__()
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, True, True]
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=1, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=1, dilate=replace_stride_with_dilation[2])
        self.aspp = ASPP(512 * Bottleneck.expansion, 256)

        # ViT 部分
        self.vit_dim = vit_dim
        self.vit_proj = nn.Linear(256, vit_dim)  # 投影到 vit_dim
        self.vit_pos_embed = nn.Parameter(torch.zeros(1, 784, vit_dim))  # 假设特征图尺寸为 28x28
        self.vit_blocks = nn.ModuleList([
            ViTBlock(dim=vit_dim, num_heads=vit_heads, mlp_ratio=vit_mlp_ratio) for _ in range(vit_depth)
        ])
        self.vit_norm = nn.LayerNorm(vit_dim)
        self.fc = nn.Linear(vit_dim, num_classes)
        self._initialize_weights()

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = 1
        dilation = previous_dilation
        if dilate:
            dilation *= stride
            stride = 1
        else:
            dilation = 1
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.in_planes, planes, stride, dilation, downsample)]
        self.in_planes = planes * block.expansion
        layers.extend(block(self.in_planes, planes, dilation=dilation) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        nn.init.trunc_normal_(self.vit_pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
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
        x = self.aspp(x)

        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        x = self.vit_proj(x)  # 将通道数映射到 vit_dim

        # 动态生成vit_pos_embed
        num_patches = H * W
        vit_pos_embed = self.vit_pos_embed[:, :num_patches, :].to(x.device)
        x = x + vit_pos_embed  # 添加位置嵌入
        x = x.transpose(0, 1)  # 转换形状用于ViT

        for blk in self.vit_blocks:
            x = blk(x)

        x = self.vit_norm(x)
        x = x.mean(dim=0)  # 将所有 patch 平均得到图像表示
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

    # 创建模型实例
    model = ResNet50_Dilated_ASPP_ViT(
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
