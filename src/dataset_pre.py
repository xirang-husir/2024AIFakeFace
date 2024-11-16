# -*- coding: utf-8 -*-
# @Time    : 2024/11/8 10:02
# @Author  : Husir
# @File    : dataset_pre.py
# @Software: PyCharm
# @Description : 数据集预处理，数据集合并，分割训练集，验证集，测试集
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

# 合并数据集
def merge_label_csvs(csv_files, output_csv):
    # 读取没有标题的 CSV 文件，并设置列名为 "ImageName" 和 "Label"
    combined_df = pd.concat(
        (pd.read_csv(f, header=None, names=["ImageName", "Label"]) for f in csv_files),
        ignore_index=True
    )
    # 保存为新的 CSV 文件
    combined_df.to_csv(output_csv, index=False)

# 定义三个 CSV 文件的路径和输出文件路径
csv_files = ["./dataset_divid/test/labels/test_labels.csv", "./dataset_divid/train/labels/train_labels.csv"]
output_csv = "./dataset_divid/test_train/labels/test_labels.csv"

merge_label_csvs(csv_files, output_csv)

# # 划分数据集
#
# # 合并后的数据集 CSV 文件路径
# combined_csv_path = "./dataset/dataset.csv"
# # 合并后图片数据集所在目录
# source_image_dir = "./dataset/data"
# # 输出文件夹路径
# output_dir = "./dataset_divid"
# # 子目录名称
# train_dir = os.path.join(output_dir, "train")
# test_dir = os.path.join(output_dir, "test")
# valid_dir = os.path.join(output_dir, "valid")
#
# # 创建输出目录结构
# os.makedirs(os.path.join(train_dir, "images"), exist_ok=True)
# os.makedirs(os.path.join(train_dir, "labels"), exist_ok=True)
# os.makedirs(os.path.join(test_dir, "images"), exist_ok=True)
# os.makedirs(os.path.join(test_dir, "labels"), exist_ok=True)
# os.makedirs(os.path.join(valid_dir, "images"), exist_ok=True)
# os.makedirs(os.path.join(valid_dir, "labels"), exist_ok=True)
#
#
# def find_image_path(image_name):
#     """
#     检查是否存在扩展名匹配的文件
#     """
#     possible_extensions = ['.jpg', '.jpeg', '.png']
#     for ext in possible_extensions:
#         img_path = os.path.join(source_image_dir, image_name + ext)
#         if os.path.exists(img_path):
#             return img_path
#     return None
#
#
# def split_and_save_dataset(csv_path):
#     # 读取合并后的数据集
#     combined_df = pd.read_csv(csv_path, header=None, names=["ImageName", "Label"])
#
#     # 第一次划分：8:2 划分为 train 和 test
#     train_df, test_df = train_test_split(combined_df, test_size=0.2, random_state=42)
#
#     # 第二次划分：将 train 按 9:1 划分为 train 和 valid
#     train_df, valid_df = train_test_split(train_df, test_size=0.1, random_state=42)
#
#     # 定义保存函数
#     def save_images_and_labels(data_df, dest_dir):
#         images_dir = os.path.join(dest_dir, "images")
#         labels_file_path = os.path.join(dest_dir, "labels", f"{os.path.basename(dest_dir)}_labels.csv")
#
#         # 创建标签文件
#         with open(labels_file_path, 'w') as label_file:
#             for _, row in data_df.iterrows():
#                 img_name, label = row["ImageName"], row["Label"]
#                 # 写入标签文件
#                 label_file.write(f"{img_name},{label}\n")
#
#                 # 检查是否存在图像文件
#                 img_src_path = find_image_path(img_name)
#                 if img_src_path:
#                     img_dest_path = os.path.join(images_dir, os.path.basename(img_src_path))
#                     shutil.copy(img_src_path, img_dest_path)
#                 else:
#                     print(f"Warning: Image for {img_name} not found with any common extension.")
#
#     # 保存 train, valid, test 数据集
#     save_images_and_labels(train_df, train_dir)
#     save_images_and_labels(valid_df, valid_dir)
#     save_images_and_labels(test_df, test_dir)
#
#
# # 执行数据集划分和保存
# split_and_save_dataset(combined_csv_path)
# print("数据集划分完成并保存。")


