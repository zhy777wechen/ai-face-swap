# -*- coding: utf-8 -*-
"""
频域伪影特征。

对应 PPT「Deepfake 检测：频域伪影——DCT/FFT 捕捉上采样棋盘格痕迹」。
几何换脸同样存在重采样：src 的对齐 warp 与色彩校正都会引入插值痕迹，
这类痕迹在空域几乎不可见，但在频域表现为周期性能量集中。

两路特征拼成多通道输入：
  1. DCT  —— 分块 8x8 DCT（JPEG 域）后取 log 幅度，能放大压缩/上采样造成的块效应差异
  2. FFT  —— 整幅 log 幅度谱减均值（去直流），保留高频段的能量分布

约定：所有归一化只用当前图自身的统计量，不引入数据集级别的均值方差，
避免训练/推理的预处理不一致，也便于第 04 阶段直接在此之上加扰动。
"""

import cv2
import numpy as np

from . import config

EPS = 1e-6


def _to_gray01(im):
    if im is None:
        raise ValueError("输入图像为 None")
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im.astype(np.float32) / 255.0


def _zscore(mat):
    std = float(mat.std())
    if std < EPS:
        return np.zeros_like(mat)
    return (mat - mat.mean()) / (std + EPS)


def dct_channels(gray_im, size=config.FREQ_SIZE, blocks=((8, 8), (16, 16))):
    """分块 DCT 的 log 幅度图。blocks 给出两组块尺寸 (高, 宽)，覆盖细/粗两种网格。

    返回 (len(blocks), size, size) 的 float32，取值近似 z-score。
    """
    im = cv2.resize(_to_gray01(gray_im), (size, size), interpolation=cv2.INTER_LINEAR)
    out = []
    for bh, bw in blocks:
        nby, nbx = size // bh, size // bw
        acc = np.zeros((nby, nbx), np.float32)
        for by in range(nby):
            for bx in range(nbx):
                blk = im[by * bh:(by + 1) * bh, bx * bw:(bx + 1) * bw]
                if blk.shape[0] != bh or blk.shape[1] != bw:
                    continue
                d = cv2.dct(np.ascontiguousarray(blk))
                acc[by, bx] = np.log1p(np.abs(d)).mean()
        acc = cv2.resize(acc, (size, size), interpolation=cv2.INTER_LINEAR)
        out.append(_zscore(acc))
    return np.stack(out, axis=0)


def fft_channel(gray_im, size=config.FREQ_SIZE, high_pass_ratio=0.25):
    """整幅 FFT 的 log 幅度谱（去直流、只保留高通部分），输出 size x size。

    低能量区被置零，让网络专注高频结构差异。
    """
    im = cv2.resize(_to_gray01(gray_im), (size, size), interpolation=cv2.INTER_LINEAR)
    full = cv2.copyMakeBorder(im, 0, size, 0, size, cv2.BORDER_CONSTANT)
    spec = cv2.dft(np.float32(full), flags=cv2.DFT_COMPLEX_OUTPUT)
    shift = np.roll(spec, (size, size), axis=(0, 1))
    big = size * 2
    mag = np.log1p(cv2.magnitude(shift[:, :, 0], shift[:, :, 1]))
    mag = _zscore(mag)
    if 0.0 < high_pass_ratio < 0.5:
        mask = np.ones_like(mag)
        cy, cx = big // 2, big // 2          # roll 之后直流分量在正中
        r = int(big * high_pass_ratio)
        yy, xx = np.ogrid[:big, :big]
        mask[((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r] = 0.0
        mag = mag * mask
    mag = cv2.resize(mag, (size, size), interpolation=cv2.INTER_AREA)
    return mag[np.newaxis, ...].astype(np.float32)


def frequency_features(bgr_im, size=config.FREQ_SIZE):
    """把 BGR 人脸裁剪转成频域分支输入：(C, FREQ_SIZE, FREQ_SIZE) float32。"""
    gray = cv2.cvtColor(bgr_im, cv2.COLOR_BGR2GRAY) if bgr_im.ndim == 3 else bgr_im
    feats = [dct_channels(gray), fft_channel(gray)]
    return np.concatenate(feats, axis=0).astype(np.float32)
