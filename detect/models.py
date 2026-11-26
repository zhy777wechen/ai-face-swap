# -*- coding: utf-8 -*-
"""
检测网络结构。

1. LivenessCNN —— PPT 第 10 页“活体检测”那一列的逐层复刻：
       Conv3x3-16 -> ReLU -> MaxPool -> Conv3x3-32 -> ReLU -> MaxPool
       -> FC128 -> ReLU -> FC2
   刻意不额外加 BN / Dropout / 更深的主干，目的是让“课程基线结构”的指标
   可解释、可复现，也为第 04 阶段的 FGSM/PGD 提供干净的攻击面。

2. DualBranchNet —— 空域分支（同一个特征提取器）+ 频域分支（DCT/FFT 输入），
   两支特征拼接后再融合分类。对应 PPT 第 13 页“构建空域加频域双分支检测网络”。

权重读写用 save_checkpoint / load_checkpoint：结构参数与预处理配置一起存盘，
推理与对抗实验只需指定 .pt 文件即可还原，不必手工对齐命令行参数。
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config

# 空域输入的归一化：uint8 -> [-1, 1]
SPATIAL_MEAN = 0.5
SPATIAL_STD = 0.5


def spatial_feature_size(img_size):
    """两层 3x3 卷积 + 两次 2x2 最大池化后的展平长度。"""
    side = img_size // 4
    if side * 4 != img_size:
        raise ValueError(f"img_size 需要能被 4 整除，当前为 {img_size}")
    return 32 * side * side


class SpatialEncoder(nn.Module):
    """PPT 结构里 FC 之前的部分：Conv3x3-16 -> ReLU -> MaxPool -> Conv3x3-32 -> ReLU -> MaxPool。"""

    def __init__(self, in_channels=3, img_size=config.IMG_SIZE):
        super().__init__()
        self.in_channels = in_channels
        self.img_size = img_size
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),   # 与 PPT 的 3x3-16 对齐
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),            # 3x3-32
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.out_dim = spatial_feature_size(img_size)

    def forward(self, x):
        return self.body(x).flatten(1)


class LivenessCNN(nn.Module):
    """活体检测基线：真伪二分类，结构完全按 PPT。"""

    def __init__(self, in_channels=3, img_size=config.IMG_SIZE, num_classes=2):
        super().__init__()
        self.encoder = SpatialEncoder(in_channels, img_size)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.out_dim, 128),                   # FC128
            nn.ReLU(),
            nn.Linear(128, num_classes),                            # FC2
        )

    def forward(self, x):
        return self.head(self.encoder(x))


class FrequencyEncoder(nn.Module):
    """频域分支：输入 frequency_features() 的输出（log-DCT / log-FFT，已 z-score）。

    结构与空域分支同量级（16、32 两层 3x3 + 池化），差别在于用 4x4 全局平均池化
    收束尺寸——频域图的空间位置含义弱，重点是能量分布，不必保留高分辨率展平。
    """

    def __init__(self, in_channels=3, freq_size=config.FREQ_SIZE):
        super().__init__()
        self.in_channels = in_channels
        self.freq_size = freq_size
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.out_dim = 32 * 4 * 4

    def forward(self, x):
        return self.body(x).flatten(1)


class DualBranchNet(nn.Module):
    """空域 + 频域双分支；freq_only=False 时退化为只用空域支。"""

    def __init__(self, img_size=config.IMG_SIZE, freq_size=config.FREQ_SIZE,
                 spatial_channels=3, freq_channels=3, num_classes=2, dropout=0.2):
        super().__init__()
        self.spatial = SpatialEncoder(spatial_channels, img_size)
        self.freq = FrequencyEncoder(freq_channels, freq_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.spatial.out_dim + self.freq.out_dim, 128)
        self.out = nn.Linear(128, num_classes)

    def forward(self, spatial, freq):
        f = torch.cat([self.spatial(spatial), self.freq(freq)], dim=1)
        f = self.dropout(F.relu(self.fc(f)))
        return self.out(f)


def build_model(arch="dual", img_size=config.IMG_SIZE, freq_size=config.FREQ_SIZE,
                freq_channels=3, num_classes=2):
    """arch: 'liveness' 单分支基线，'dual' 空域+频域双分支。"""
    if arch == "liveness":
        return LivenessCNN(img_size=img_size, num_classes=num_classes)
    if arch == "dual":
        return DualBranchNet(img_size=img_size, freq_size=freq_size,
                             freq_channels=freq_channels, num_classes=num_classes)
    raise ValueError(f"未知结构: {arch}（可选 liveness / dual）")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(path, model, model_meta, extra=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "model_meta": model_meta,          # arch / img_size / freq_size / freq_channels ...
        "class_names": list(config.CLASS_NAMES),
    }
    if extra:
        payload.update(extra)             # 训练指标、最优阈值、数据集路径等
    torch.save(payload, path)
    return path


def load_checkpoint(path, device=None, map_to_cpu=True):
    """返回 (model, meta)。结构由存盘时的 model_meta 还原，避免参数不一致。"""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到权重文件: {path}")
    ckpt = torch.load(path, map_location="cpu" if map_to_cpu else device, weights_only=False)
    meta = dict(ckpt.get("model_meta", {}))
    model = build_model(arch=meta.get("arch", "dual"),
                        img_size=meta.get("img_size", config.IMG_SIZE),
                        freq_size=meta.get("freq_size", config.FREQ_SIZE),
                        freq_channels=meta.get("freq_channels", 3),
                        num_classes=meta.get("num_classes", 2))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device if device is not None else torch.device("cpu"))
    model.eval()
    return model, meta, ckpt


def softmax_prob_fake(logits):
    """取“伪造”类的概率。logits 可为 numpy 或 torch。"""
    if isinstance(logits, np.ndarray):
        x = torch.from_numpy(logits.astype("float32"))
    else:
        x = logits.detach()
    return F.softmax(x, dim=-1)[:, 1].cpu().numpy()
