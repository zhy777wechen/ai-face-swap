# -*- coding: utf-8 -*-
"""
manifest.csv 读写与概况统计。

单独成模块的原因：生成数据集（build_dataset.py）不应该依赖 PyTorch。
读图、切样本、写 csv 这些都属于数据管理，与训练框架解耦后，
在只有 numpy/opencv 的机器上也能准备数据，再到 GPU 机器上训练。

列约定（config.MANIFEST_COLUMNS）：
    path          相对数据集根目录的 jpg 路径
    label         0=真实 / 1=伪造
    group         身份组；切分以它为单位（同一底图的真/假样本必须同侧）
    generator     产出方式，如 swap:dlib / synthetic，便于分组看指标
    source_image  溯源用的原始文件名（伪造样本记 “底图+源脸”）
"""

import csv
import os
from collections import Counter

from . import config


def read_manifest(data_dir):
    """读 manifest.csv，返回 list[dict]；path 已拼成绝对路径。"""
    path = os.path.join(data_dir, config.MANIFEST_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到 {path}，请先运行 python detect/build_dataset.py 生成数据集")
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("path"):
                continue
            rows.append({
                "path": os.path.join(data_dir, row["path"].replace("/", os.sep)),
                "label": int(row["label"]),
                "group": row.get("group") or os.path.splitext(os.path.basename(row["path"]))[0],
                "generator": row.get("generator", ""),
                "source_image": row.get("source_image", ""),
            })
    if not rows:
        raise ValueError(f"{path} 里没有可用样本")
    return rows


def write_manifest(data_dir, rows):
    """写出 manifest.csv（相对路径），返回文件路径。"""
    path = os.path.join(data_dir, config.MANIFEST_NAME)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=config.MANIFEST_COLUMNS)
        w.writeheader()
        for r in rows:
            item = dict(r)
            if os.path.isabs(str(item.get("path", ""))):
                item["path"] = os.path.relpath(item["path"], data_dir)
            item["path"] = str(item["path"]).replace(os.sep, "/")
            w.writerow({k: item.get(k, "") for k in config.MANIFEST_COLUMNS})
    return path


def summarize(rows):
    """样本数、真伪比、身份组数、生成方式分布。"""
    labels = [r["label"] for r in rows]
    return {
        "total": len(rows),
        "real": int(labels.count(config.LABEL_REAL)),
        "fake": int(labels.count(config.LABEL_FAKE)),
        "groups": len({r["group"] for r in rows}),
        "generators": dict(Counter(r["generator"] or "-" for r in rows)),
    }
