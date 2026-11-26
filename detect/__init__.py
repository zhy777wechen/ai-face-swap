# -*- coding: utf-8 -*-
"""
检测模块（课程实施计划第 03 阶段）

对应 PPT「防御路径 / 检测模块」：
  1. 活体检测 CNN          —— models.LivenessCNN，结构与 PPT 第 10 页一致
                             Conv3x3-16 -> ReLU -> MaxPool -> Conv3x3-32
                             -> ReLU -> MaxPool -> FC128 -> ReLU -> FC2
  2. 空域 + 频域双分支网络  —— models.DualBranchNet + freq_features.FrequencyEncoder
  3. 数据集评估            —— train.py / evaluate.py，输出 ACC / F1 / AUC / EER / 混淆矩阵

目录结构（全部为本模块新增文件，src/ 下的攻击侧源码未做任何改动）：
    config.py          公共常量与默认值
    image_io.py        中文路径读写图（沿用 src 中已验证的做法）
    face_crop.py       人脸检测 + 对齐裁剪（dlib 优先，OpenCV Haar 兜底）
    freq_features.py   DCT / FFT 频域伪影特征
    metrics.py         纯 numpy 实现的 AUC / EER / 混淆矩阵
    models.py          两种网络结构与权重读写
    manifest.py        manifest.csv 读写（不依赖 torch，供 build_dataset 复用）
    dataset.py         按身份分组切分、增强、DataLoader 数据集
    inference.py       按权重 meta 复现预处理，train/evaluate/detect 共用
    build_dataset.py   自建数据集生成器（复用 src 的换脸流水线）
    train.py           训练入口
    evaluate.py        评估入口
    detect.py          单图 / 批量推理入口

依赖：torch（见 detect/requirements.txt）。
本模块只做检测；FGSM / PGD 对抗实验属于第 04 阶段，接口上通过
models.load_checkpoint() 复用此处权重，无需改动检测代码。
"""
