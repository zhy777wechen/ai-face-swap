# -*- coding: utf-8 -*-
"""
数据集组织与切分。

约定：自建数据集目录布局（build_dataset.py 产出）
    detect_data/
        manifest.csv          列：path,label,group,generator,source_image
        real/xxx.jpg          真实样本（原始照片的人脸裁剪）
        fake/xxx.jpg          伪造样本（换脸结果的人脸裁剪）
    path 为相对数据集根目录的路径。

切分按 group 分组（GroupShuffleSplit 语义，自己实现以免依赖 sklearn）：
同一个身份/同一张底图衍生的样本必须落在同一个 split。否则“真实照片 A 的原图”
与“以 A 为底图的换脸结果”会分别进入训练集与验证集，网络可以凭背景纹理、
光照而非伪造痕迹记住身份，验证指标会虚高——这是本类数据集最容易踩的坑。
"""

import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from . import config
from .image_io import imread_unicode
from .manifest import read_manifest, summarize, write_manifest  # re-export，兼容 `ds.read_manifest`
from .models import SPATIAL_MEAN, SPATIAL_STD


def group_split(rows, val_ratio=config.VAL_RATIO, test_ratio=config.TEST_RATIO, seed=config.SEED):
    """按 group 切 train/val(/test)。test_ratio<=0 时返回 test=None。"""
    groups = sorted({r["group"] for r in rows})
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    groups = [groups[i] for i in order]

    n_val = max(1, int(round(len(groups) * val_ratio)))
    n_test = max(0, int(round(len(groups) * test_ratio))) if test_ratio > 0 else 0
    if n_val + n_test >= len(groups):
        n_test = max(0, len(groups) - n_val - 1)

    val_groups = set(groups[:n_val])
    test_groups = set(groups[n_val:n_val + n_test]) if n_test else set()
    train, val, test = [], [], []
    for r in rows:
        if r["group"] in val_groups:
            val.append(r)
        elif r["group"] in test_groups:
            test.append(r)
        else:
            train.append(r)
    if not train or not val:
        raise ValueError(f"分组切分后 train={len(train)} val={len(val)}，"
                         f"请增加不同身份/底图的样本数")
    return train, val, (test or None), {"val_groups": sorted(val_groups),
                                        "test_groups": sorted(test_groups)}


def balance_labels(rows, target=None, seed=config.SEED):
    """把多数类下采样到与少数类相当，避免类别不平衡时 ACC 虚高。"""
    real = [r for r in rows if r["label"] == config.LABEL_REAL]
    fake = [r for r in rows if r["label"] == config.LABEL_FAKE]
    if not real or not fake:
        return rows
    n = min(len(real), len(fake)) if target is None else int(target)
    n = max(1, min(n, len(real), len(fake)))
    rng = np.random.default_rng(seed)
    ri = rng.choice(len(real), size=n, replace=False)
    fi = rng.choice(len(fake), size=n, replace=False)
    return [real[i] for i in sorted(ri)] + [fake[i] for i in sorted(fi)]


class FaceForgeryDataset(Dataset):
    """产出 {spatial, freq, label, index}；use_freq=False 时 freq 为 None。

    几何增强（水平翻转、小平移）作用在 uint8 裁剪图上，两个分支用同一张变换后的图，
    保证空域与频域看到的始终是同一个样本。默认不做亮度/对比度扰动——
    那类光度改动会直接破坏我们要利用的频域伪影。
    """

    def __init__(self, rows, img_size=config.IMG_SIZE, freq_size=config.FREQ_SIZE,
                 use_freq=True, augment=False, freq_channels=3):
        self.rows = list(rows)
        self.img_size = img_size
        self.freq_size = freq_size
        self.use_freq = use_freq
        self.augment = augment
        self.freq_channels = freq_channels

    def __len__(self):
        return len(self.rows)

    def _load(self, row):
        im = imread_unicode(row["path"])
        if im is None:
            raise FileNotFoundError(f"样本读取失败: {row['path']}")
        return im

    def _augment(self, im):
        h, w = im.shape[:2]
        if self.augment and self.rng.random() < 0.5:
            im = cv2.flip(im, 1)
        if self.augment:
            dx = int(self.rng.integers(-w // 16, w // 16 + 1))
            dy = int(self.rng.integers(-h // 16, h // 16 + 1))
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            im = cv2.warpAffine(im, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return im

    def set_rng(self, epoch):
        self.rng = np.random.default_rng(config.SEED + int(epoch))

    def __getitem__(self, i):
        if not hasattr(self, "rng"):
            self.set_rng(0)
        row = self.rows[i]
        im = self._augment(self._load(row))

        spatial = cv2.resize(im, (self.img_size, self.img_size),
                             interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        spatial = (spatial - SPATIAL_MEAN) / SPATIAL_STD
        spatial = np.transpose(spatial, (2, 0, 1))

        freq = None
        if self.use_freq:
            from . import freq_features
            f = freq_features.frequency_features(im, size=self.freq_size)
            if f.shape[0] != self.freq_channels:
                f = f[:self.freq_channels]
            freq = f.astype(np.float32)

        return {
            "spatial": torch.from_numpy(np.ascontiguousarray(spatial)),
            "freq": torch.from_numpy(np.ascontiguousarray(freq)) if freq is not None else None,
            "label": torch.tensor(row["label"], dtype=torch.long),
            "index": torch.tensor(i, dtype=torch.long),
        }


def collate_batch(batch):
    out = {
        "spatial": torch.stack([b["spatial"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "index": torch.stack([b["index"] for b in batch]),
    }
    freqs = [b["freq"] for b in batch]
    out["freq"] = torch.stack(freqs) if freqs[0] is not None else None
    return out
