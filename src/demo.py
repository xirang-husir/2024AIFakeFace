# -*- coding: utf-8 -*-
# @Time    : 2024/11/8 15:19
# @Author  : Husir
# @File    : demo.py
# @Software: PyCharm
# @Description : 遍历数据集，求出尺寸的最大和最小
import os
from PIL import Image

# def find_max_min_image_sizes(image_dir):
#     max_size = [0, 0]  # 初始化最大尺寸 [宽, 高]
#     min_size = [float('inf'), float('inf')]  # 初始化最小尺寸 [宽, 高]
#     count = 0  # 统计图片数量
#     for filename in os.listdir(image_dir):
#         if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
#             count += 1
#             image_path = os.path.join(image_dir, filename)
#             with Image.open(image_path) as img:
#                 width, height = img.size
#                 # 更新最大尺寸
#                 if height > max_size[1] or (height == max_size[1] and width > max_size[0]):
#                     max_size = [width, height]
#                 # 更新最小尺寸
#                 if height < min_size[1] or (height == min_size[1] and width < min_size[0]):
#                     min_size = [width, height]
#
#     return max_size, min_size, count
#
# # 指定图片目录
# image_dir = "./dataset/data"
# max_size, min_size, image_count = find_max_min_image_sizes(image_dir)
#
# print(f"最大尺寸: {max_size}")
# print(f"最小尺寸: {min_size}")
# print(f"图片数量: {image_count}")

from PIL import Image
import cv2

# 加载图像并检测人脸
# img = cv2.imread('./dataset/data/tY8IVSKHzkzr31Vs.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#
# # 绘制检测到的人脸
# for (x, y, w, h) in faces:
#     cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#
# # 将图像从 BGR 转换为 RGB 格式，然后用 PIL 显示
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# pil_img = Image.fromarray(img_rgb)
# pil_img.show()


import os
import pandas as pd

# # 定义图像目录和标签文件路径
# image_dir = './dataset_divid/test/images/'
# label_file = './dataset_divid/test/labels/test_labels.csv'
#
# # 读取标签文件中的文件名列（假设文件名在第一列）
# labels_df = pd.read_csv(label_file, header=None)
# label_image_names = set(labels_df[0].apply(lambda x: x.split('.')[0]))  # 去掉扩展名
#
# # 获取图像目录中的所有图像文件名（不含扩展名）
# image_files = set(os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
#
# # 找出标签文件中有但图像目录中没有的文件名
# missing_images = label_image_names - image_files
#
# # 找出图像目录中有但标签文件中没有的文件名
# extra_images = image_files - label_image_names
#
# # 输出检查结果
# if missing_images:
#     print(f"标签文件中的图像在图像目录中找不到: {missing_images}")
# else:
#     print("标签文件中的所有图像在图像目录中都存在。")
#
# if extra_images:
#     print(f"图像目录中的图像在标签文件中找不到: {extra_images}")
# else:
#     print("图像目录中的所有图像在标签文件中都存在。")


def analyze_image_sizes(dataset_path, max_dim=1024):
    # 初始化最小、最大尺寸和计数器
    min_size = None
    max_size = None
    count_larger_than_max = 0

    # 遍历数据集目录中的所有文件
    for filename in os.listdir(dataset_path):
        filepath = os.path.join(dataset_path, filename)

        try:
            # 打开图片并获取尺寸
            with Image.open(filepath) as img:
                width, height = img.size
                img_size = (width, height)

                # 更新最小尺寸
                if min_size is None or (width * height < min_size[0] * min_size[1]):
                    min_size = img_size

                # 更新最大尺寸
                if max_size is None or (width * height > max_size[0] * max_size[1]):
                    max_size = img_size

                # 检查是否有边长超过指定的最大尺寸
                if max(width, height) > max_dim:
                    count_larger_than_max += 1

        except IOError:
            print(f"无法打开文件 {filename}，请检查文件格式是否正确")

    # 输出结果
    print(f"最小图片尺寸: {min_size[0]} x {min_size[1]}")
    print(f"最大图片尺寸: {max_size[0]} x {max_size[1]}")
    print(f"边长大于 {max_dim} 的图片数量: {count_larger_than_max}")

# 设置图片数据集路径
dataset_path = "E:/DownloadFromEdge/dataset/data"
analyze_image_sizes(dataset_path)

