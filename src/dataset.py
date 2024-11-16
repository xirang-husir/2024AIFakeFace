# -*- coding: utf-8 -*-
# @Time    : 2024/11/7 16:20
# @Author  : Husir
# @File    : dataset_img.py
# @Software: PyCharm
# @Description : 数据集预处理
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        # 读取标签
        self.labels_df = pd.read_csv(label_file,header=None)
        # 使用列索引代替列名
        self.images = list(zip(self.labels_df[0], self.labels_df[1]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name, label = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # 尝试不同的扩展名
        possible_extensions = ['.jpg', '.jpeg', '.png']
        if not img_path.lower().endswith(tuple(possible_extensions)):
            for ext in possible_extensions:
                img_path_with_ext = img_path + ext
                if os.path.exists(img_path_with_ext):
                    img_path = img_path_with_ext
                    break

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
#创建测试数据集类


class TestImageDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        # 读取标签，并确保按字典序排序文件名
        self.image_list = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        self.labels_df = pd.read_csv(label_file, header=None)
        self.images = list(zip(self.labels_df[0], self.labels_df[1]))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name, label = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # 如果 img_path 没有扩展名，尝试附加可能的扩展名进行检查
        possible_extensions = ['.jpg', '.jpeg', '.png']
        if not img_path.lower().endswith(tuple(possible_extensions)):
            for ext in possible_extensions:
                img_path_with_ext = img_path + ext
                if os.path.exists(img_path_with_ext):
                    img_path = img_path_with_ext
                    break

        # 确保图片存在
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")

        # 打开图片并应用变换
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # 去除文件扩展名
        img_id = os.path.splitext(img_name)[0]

        return image, img_id, int(label)  # 确保 label 为 int

# 定义数据集变换函数
class ResizeWithAspectRatio:
    def __init__(self, max_size=1024):
        """
        初始化，设定图像的最大尺寸限制。
        :param max_size: 图像的最大尺寸上限，确保不超过该尺寸。
        """
        self.max_size = max_size

    def __call__(self, img):
        # 获取图像的原始宽高
        w, h = img.size

        # 计算最大边长度，并判断是否需要缩放
        if max(w, h) > self.max_size:
            # 计算缩放比例，确保长宽比不变
            scaling_factor = self.max_size / max(w, h)
            new_w, new_h = int(w * scaling_factor), int(h * scaling_factor)
            img = img.resize((new_w, new_h), Image.ANTIALIAS)  # 使用抗锯齿缩放
        return img

