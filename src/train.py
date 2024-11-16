# -*- coding: utf-8 -*-
# @Time    : 2024/11/7
# @Author  : Husir
# @File    : train.py
# @Software: PyCharm
# @Description : 训练和验证函数，支持分布式训练、混合精度训练、TensorBoard 记录、日志保存以及保存最佳模型

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR  # 修改此行
from torch.utils.tensorboard import SummaryWriter
from resnet50_ASPPCrossScanViT import ResNet50_Dilated_ASPPCrossScanViT  # 从 resnet50_ASPPCrossScanViT.py 导入模型类
from dataset import ImageDataset, ResizeWithAspectRatio  # 从 dataset_img.py 导入数据集类
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
import time  # 导入时间模块
import torch.distributed as dist
from tqdm import tqdm  # 导入 tqdm 库

# 设置日志记录
def setup_logging(log_dir):
    """
    设置日志记录文件，将日志信息保存到指定目录中
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'training.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(log_file),  # 日志写入文件
            logging.StreamHandler()         # 日志输出到控制台
        ]
    )
    return logging.getLogger()

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, logger, writer, txt_log_path, args):
    """
    训练模型函数，包含早停机制
    """
    best_val_acc = 0.0  # 初始化最佳验证准确率
    no_improve_epochs = 0  # 记录验证集准确率未提升的 epoch 数

    for epoch in range(1, epochs + 1):
        # 记录开始时间
        start_time = time.time()

        model.train()  # 切换到训练模式
        running_loss = 0.0  # 记录本轮训练损失
        # 添加进度条
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} - Training", leave=False)
        for images, labels in train_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()  # 清除梯度

            # 前向传播与损失计算
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            running_loss += loss.item() * images.size(0)  # 累积损失
            # 更新进度条的显示信息
            train_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)  # 计算本轮平均损失

        # 验证模型性能
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

        # 更新学习率调度器
        scheduler.step()  # 修改此行

        # 记录结束时间并计算本 epoch 耗时
        end_time = time.time()
        epoch_duration = end_time - start_time

        # 仅在主进程（rank == 0）上记录日志、TensorBoard 和 txt 文件
        if args.rank == 0:
            logger.info(f"Epoch [{epoch}/{epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.6f}, Time: {epoch_duration:.2f} sec")
            writer.add_scalar('Loss/Train', epoch_loss, epoch)
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            writer.add_scalar('Accuracy/Validation', val_acc, epoch)
            writer.add_scalar('Time/Epoch', epoch_duration, epoch)

            # 将指标写入 txt 文件
            with open(txt_log_path, "a") as f:
                f.write(f"Epoch {epoch}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.6f}, Time: {epoch_duration:.2f} sec\n")

            # 检查是否保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improve_epochs = 0  # 重置计数器
                torch.save(model.state_dict(), args.model_save_path)
                logger.info(f"保存最佳模型到 {args.model_save_path}")
                writer.add_scalar('Best Val Accuracy', best_val_acc, epoch)
            else:
                no_improve_epochs += 1  # 增加未提升计数

            # 检查早停条件
            if no_improve_epochs >= args.early_stop_patience:
                logger.info(f"验证准确率在 {args.early_stop_patience} 个 epoch 未提升，提前停止训练。")
                break

    if args.rank == 0:
        logger.info("训练完成。")


def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    # 添加验证进度条
    val_bar = tqdm(val_loader, desc="Validating", leave=False)
    with torch.no_grad():
        for images, labels in val_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # 更新进度条的显示信息
            val_bar.set_postfix(loss=loss.item())

    # 汇总各进程的损失和准确率，仅在分布式情况下进行
    if dist.is_initialized() and dist.get_world_size() > 1:
        local_val_loss = torch.tensor([running_loss], device=device)
        local_total = torch.tensor([total], device=device)
        local_correct = torch.tensor([correct], device=device)

        dist.reduce(local_val_loss, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(local_total, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(local_correct, dst=0, op=dist.ReduceOp.SUM)

        if dist.get_rank() == 0:
            val_loss = local_val_loss.item() / local_total.item()
            val_acc = local_correct.item() / local_total.item()
        else:
            val_loss = None
            val_acc = None
    else:
        # 如果不是分布式环境，直接计算结果
        val_loss = running_loss / total
        val_acc = correct / total

    return val_loss, val_acc


# 参数解析
def parse_args():
    parser = argparse.ArgumentParser(description='训练 ResNet50 + Dilated Conv 模型')
    parser.add_argument('--num_classes', type=int, default=2, help='分类任务的类别数量')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='训练和验证的批量大小')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='优化器的权重衰减')
    parser.add_argument('--model_save_path', type=str, default='./results/ResNet50_Dilated_ASPPCrossScanViT2/best_epoch_model.pth', help='最佳模型保存路径')
    parser.add_argument('--train_label_file', type=str, default='./dataset_divid/test_train/labels/test_train.csv', help='训练标签 CSV 文件路径')
    parser.add_argument('--val_label_file', type=str, default='./dataset_divid/valid/labels/valid_labels.csv', help='验证标签 CSV 文件路径')
    parser.add_argument('--train_image_dir', type=str, default='./dataset_divid/test_train/images', help='训练集图像目录')
    parser.add_argument('--val_image_dir', type=str, default='./dataset_divid/valid/images', help='验证集图像目录')
    parser.add_argument('--log_dir', type=str, default='./tensorboard/ResNet50_Dilated_ASPPCrossScanViT2', help='日志和 TensorBoard 文件保存目录')
    parser.add_argument('--txt_log_path', type=str, default='./results/ResNet50_Dilated_ASPPCrossScanViT2/training_metrics.txt', help='训练指标保存路径')
    parser.add_argument('--dist_backend', type=str, default='nccl', help='分布式训练的后端')
    parser.add_argument('--dist_url', type=str, default='env://', help='用于设置分布式训练的 URL')
    parser.add_argument('--world_size', type=int, default=1, help='参与训练的进程数（GPU数）')
    parser.add_argument('--rank', type=int, default=0, help='当前进程的编号')
    parser.add_argument('--early_stop_patience', type=int, default=20, help='早停的 epoch 数')
    # 添加余弦调度器相关参数
    parser.add_argument('--cosine_T_max', type=int, default=60, help='余弦调度器的 T_max 参数')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='初始学习率')
    parser.add_argument('--cosine_eta_min', type=float, default=1e-6, help='余弦调度器的最小学习率')

    args = parser.parse_args()
    return args


def main(args):
    # 初始化分布式训练（如果需要）
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
    logger.info("开始训练脚本")

    # 设置 TensorBoard
    writer = SummaryWriter(log_dir=args.log_dir)

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        ResizeWithAspectRatio(max_size=1024),  # 保持长宽比地缩放
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),  # 随机裁剪
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
        # transforms.RandomGrayscale(p=0.1),  # 随机灰度
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集
    train_dataset = ImageDataset(args.train_image_dir, args.train_label_file, transform=transform)
    val_dataset = ImageDataset(args.val_image_dir, args.val_label_file, transform=transform)
    # 分布式训练时使用 DistributedSampler
    if args.world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                              sampler=train_sampler, num_workers=12, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            sampler=val_sampler, num_workers=12, pin_memory=True)

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

    # 如果是分布式训练，封装模型
    if args.world_size > 1:
        model = DDP(model, device_ids=[args.rank])

    # 定义损失函数和优化器
    class_weights = torch.tensor([1.0] * args.num_classes).to(device)  # 类别权重
    criterion = nn.CrossEntropyLoss(weight=class_weights)  # 加权交叉熵损失 计算预测与标签之间的误差
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # 定义余弦学习率调度器
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.cosine_T_max,
        eta_min=args.cosine_eta_min
    )

    # 确保 txt 文件路径存在
    os.makedirs(os.path.dirname(args.txt_log_path), exist_ok=True)

    # 训练模型
    train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        args.epochs,
        logger,
        writer,
        args.txt_log_path,
        args  # 传递 args 以在 train_model 中使用 args.rank
    )

    # 清理分布式环境
    if args.world_size > 1:
        dist.destroy_process_group()

    writer.close()
    logger.info("训练脚本结束。")


if __name__ == "__main__":
    args = parse_args()
    # 确保模型保存路径存在
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
    # 确保日志目录存在
    os.makedirs(args.log_dir, exist_ok=True)
    main(args)
