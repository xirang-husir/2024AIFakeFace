# main.py
# -*- coding: utf-8 -*-
# @Time    : 2024/11/7
# @Author  : Husir
# @File    : main.py
# @Software: PyCharm
# @Description : 使用训练好的模型对测试集进行预测

import argparse
import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from resnet50_ASPPCrossScanViT import ResNet50_Dilated_ASPPCrossScanViT  # 从 resnet_dilatedConv.py 导入模型
from dataset import TestImageDataset, ResizeWithAspectRatio
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import logging

from tqdm import tqdm  # 导入 tqdm 库
# 设置日志记录
def setup_logging(log_dir):
    """
    设置日志记录，确保日志文件和输出格式
    :param log_dir: 日志保存目录
    :return: 日志记录器
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'testing.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(log_file),   # 日志写入文件
            logging.StreamHandler()          # 日志输出到控制台
        ]
    )
    return logging.getLogger()

# 预测函数
def predict(model, test_loader, criterion, device, logger):
    """
    使用训练好的模型对测试集进行预测，并计算准确率
    :param model: 已训练好的模型
    :param test_loader: 测试数据加载器
    :param criterion: 损失函数
    :param device: 计算设备
    :param logger: 日志记录器
    :return: 预测结果列表，每个元素为 (图像ID, 预测标签)
    """
    model.eval()
    predictions = []
    correct = 0
    total = 0
    running_loss = 0.0

    # 使用 tqdm 包裹 test_loader 添加进度条
    with torch.no_grad():
        for images, img_ids, labels in tqdm(test_loader, desc="Testing", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 收集预测结果
            for img_id, label in zip(img_ids, predicted.cpu().numpy()):
                predictions.append((img_id, int(label)))

    # 按 img_id 进行字典序排序
    predictions.sort(key=lambda x: x[0])

    accuracy = correct / total if total > 0 else 0
    avg_loss = running_loss / total if total > 0 else 0
    logger.info(f"测试集准确率: {accuracy:.6f}")
    logger.info(f"测试集平均损失: {avg_loss:.4f}")
    logger.info("预测完成。")
    return predictions





# 参数解析
def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='测试 ResNet50+ ViT + SE 模型')
    parser.add_argument('--num_classes', type=int, default=2, help='分类任务的类别数量')
    parser.add_argument('--batch_size', type=int, default=32, help='测试的批量大小')
    parser.add_argument('--model_path', type=str, default='./results/ResNet50_Dilated_ASPPCrossScanViT2/best_epoch_model.pth', help='已训练模型的路径 (.pth 文件)')
    parser.add_argument('--test_label_file', type=str, default='./dataset_divid/test/labels/test_labels.csv', help='测试标签 CSV 文件路径')
    parser.add_argument('--test_image_dir', type=str, default=r'E:\DownloadFromEdge\Kevin & Zoe20241029\src\dataset_divid\test\images', help='测试集图像目录')
    parser.add_argument('--result_path', type=str, default='./results/ResNet50_Dilated_ASPPCrossScanViT2/test_results.csv', help='预测结果保存路径')
    parser.add_argument('--log_dir', type=str, default='./logs', help='日志保存目录')
    parser.add_argument('--dist_backend', type=str, default='nccl', help='分布式的后端类型')
    parser.add_argument('--dist_url', type=str, default='env://', help='设置分布式训练的 URL')
    parser.add_argument('--world_size', type=int, default=1, help='进程总数')
    parser.add_argument('--rank', type=int, default=0, help='当前进程的编号')
    args = parser.parse_args()
    return args

def main(args):
    # 初始化分布式测试（如果需要）
    if args.world_size > 1:
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size,
                                rank=args.rank)
        torch.cuda.set_device(args.rank)
        device = torch.device(f'cuda:{args.rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 设置日志记录
    logger = setup_logging(args.log_dir)
    logger.info("开始测试脚本")

    # 数据预处理
    test_transform = transforms.Compose([
        ResizeWithAspectRatio(max_size=1024),  # 第一阶段：初始长宽比缩放
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),  # 第二阶段：随机裁剪到目标尺寸
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    # 创建测试数据集和数据加载器
    test_img_dataset = TestImageDataset(args.test_image_dir, args.test_label_file, transform=test_transform)

    # 分布式测试时使用 DistributedSampler
    if args.world_size > 1:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_img_dataset, shuffle=False)
    else:
        test_sampler = None

    test_loader = DataLoader(test_img_dataset, batch_size=args.batch_size, shuffle=False,
                             sampler=test_sampler, num_workers=10, pin_memory=True)

    # 实例化模型
    model = ResNet50_Dilated_ASPPCrossScanViT(
        num_classes=args.num_classes,
        replace_stride_with_dilation=[False, True, True],
        vit_dim=512,
        vit_depth=4,
        vit_heads=8,
        vit_mlp_ratio=4.0
    )
    model = model.to(device)

    # 如果是分布式测试，封装模型
    if args.world_size > 1:
        model = DDP(model, device_ids=[args.rank])

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 加载模型权重并移除 'module.' 前缀（如果存在）
    if os.path.exists(args.model_path):
        state_dict = torch.load(args.model_path, map_location=device)
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        logger.info(f"已加载模型权重自 {args.model_path}")
    else:
        logger.error(f"模型文件未找到: {args.model_path}")
        raise FileNotFoundError(f"模型文件未找到: {args.model_path}")

    # 获取预测结果
    predictions = predict(model, test_loader, criterion, device, logger)

    # 如果是分布式测试，仅在主进程保存结果
    if args.world_size <= 1 or (args.world_size > 1 and args.rank == 0):
        # 保存预测结果，不显示列名和索引
        result_df = pd.DataFrame(predictions, columns=['ImageID', 'Label'])
        os.makedirs(os.path.dirname(args.result_path), exist_ok=True)
        result_df.to_csv(args.result_path, header=False, index=False)
        logger.info(f"预测结果已保存到 {args.result_path}")


    # 清理分布式环境
    if args.world_size > 1:
        dist.destroy_process_group()

    logger.info("测试脚本结束。")


if __name__ == "__main__":
    args = parse_args()
    # 确保日志目录存在
    os.makedirs(args.log_dir, exist_ok=True)
    main(args)
