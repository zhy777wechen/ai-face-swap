# -*- coding: utf-8 -*-
"""检测模块的公共常量。所有 CLI 的默认值都集中在这里，便于复现实验。"""

import os

# 目录布局
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(MODULE_DIR)
SRC_DIR = os.path.join(ROOT_DIR, "src")
DATA_DIR = os.path.join(ROOT_DIR, "detect_data")      # 自建数据集默认输出
RUN_DIR = os.path.join(ROOT_DIR, "detect_runs")       # 训练权重与评估报告输出

# 样本采集
CROP_SIZE = 128          # 落盘的人脸裁剪尺寸（对齐后）
IMG_SIZE = 64            # 训练时网络输入边长（CPU 友好）
FREQ_SIZE = 128          # 频域分支的 DCT/FFT 输入边长

# 标签约定：与 PPT「真伪人脸二分类」一致
LABEL_REAL = 0           # 真实 / 活体
LABEL_FAKE = 1           # 伪造（本仓库的几何换脸产物）
CLASS_NAMES = ("real", "fake")
LABEL_CN = {LABEL_REAL: "真实", LABEL_FAKE: "伪造"}

# 数据集切分
VAL_RATIO = 0.2
TEST_RATIO = 0.0         # 默认把留出集合并进验证集；>0 时另切 test/

# 训练默认超参（PPT：算力受限时云端 GPU 训练、CPU 核心验证）
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 20261026

# manifest.csv 列顺序（build_dataset.py 写、dataset.py 读）
MANIFEST_NAME = "manifest.csv"
MANIFEST_COLUMNS = ["path", "label", "group", "generator", "source_image"]
