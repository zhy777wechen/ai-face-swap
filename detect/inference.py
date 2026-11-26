# -*- coding: utf-8 -*-
"""
推理公共层：evaluate.py 与 detect.py 都用它，保证“训练时怎么用，评估/上线就怎么用”。

关键点：预处理由权重里的 model_meta 决定，不由命令行决定。
第 04 阶段的对抗实验同样复用 iter_batches()，只需在 spatial/freq 张量上加扰动，
不必改动检测代码。
"""

import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from detect import config
    from detect.dataset import FaceForgeryDataset, collate_batch
    from detect.models import softmax_prob_fake
else:
    from . import config
    from .dataset import FaceForgeryDataset, collate_batch
    from .models import softmax_prob_fake


def pick_device(prefer=None):
    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(rows, meta, batch_size=config.BATCH_SIZE, augment=False, workers=0):
    use_freq = meta.get("arch") == "dual"
    ds = FaceForgeryDataset(rows,
                            img_size=meta.get("img_size", config.IMG_SIZE),
                            freq_size=meta.get("freq_size", config.FREQ_SIZE),
                            freq_channels=meta.get("freq_channels", 3),
                            use_freq=use_freq,
                            augment=augment)
    ds.set_rng(0)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=workers, collate_fn=collate_batch)


def iter_batches(model, meta, rows, device, batch_size=config.BATCH_SIZE, augment=False,
                 workers=0):
    """逐批产出 (logits, labels, batch_rows, input_spatial, input_freq)。

    把输入张量一并交出，是为了让对抗攻击能对同一批数据反复前向而不重新读盘。
    """
    loader = make_loader(rows, meta, batch_size, augment, workers)
    for b in loader:
        spatial = b["spatial"].to(device)
        freq = b["freq"].to(device) if b["freq"] is not None else None
        if meta.get("arch") == "dual":
            logits = model(spatial, freq)
        else:
            logits = model(spatial)
        idx = b["index"].cpu().numpy()
        yield logits, b["label"].cpu().numpy(), [rows[i] for i in idx], spatial, freq


@torch.no_grad()
def predict_rows(model, meta, rows, device=None, batch_size=config.BATCH_SIZE):
    """返回 (p_fake 数组, labels 数组)。"""
    device = device or pick_device()
    probs, labels = [], []
    for logits, y, _, _, _ in iter_batches(model, meta, rows, device, batch_size):
        probs.append(softmax_prob_fake(logits))
        labels.append(y)
    return np.concatenate(probs), np.concatenate(labels)
