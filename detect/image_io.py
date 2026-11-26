# -*- coding: utf-8 -*-
"""
中文路径读写图。

做法与 src/face_landmark_fixed.py、src/compare_three_versions.py 中已验证的一致：
部分 OpenCV 版本的 cv2.imread / cv2.imwrite 对含中文的路径返回 None 或静默失败，
因此统一走 np.fromfile + cv2.imdecode / cv2.imencode + tofile。

本模块刻意不 import src 下的脚本：src 的模块级代码会 os.chdir 到 src/ 并
立即构造 dlib detector，作为依赖引入会污染调用方的工作目录。
"""

import os

import cv2
import numpy as np

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """读取图像；中文路径失败时用 imdecode 兜底。返回 None 表示读不到。"""
    im = None
    try:
        im = cv2.imread(path, flags)
    except Exception:
        im = None
    if im is None:
        try:
            im = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
        except (OSError, ValueError):
            im = None
    return im


def imwrite_unicode(path, img, quality=92):
    """写出图像并核对真实结果，返回 (是否成功, 说明)。"""
    path = os.path.abspath(path)
    ext = os.path.splitext(path)[1]
    if not ext:
        ext = ".jpg"
        path += ext
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext.lower() in (".jpg", ".jpeg") else []
    last = ""
    try:
        if cv2.imwrite(path, img, params):
            return True, path
    except Exception as e:
        last = f"imwrite 异常: {e}"
    else:
        last = "imwrite 返回 False（权限/占用/路径问题）"

    try:
        ok, buf = cv2.imencode(ext, img, params)
        if ok:
            buf.tofile(path)
            if os.path.exists(path) and os.path.getsize(path) > 0:
                return True, path
            last += "；imencode 写入后文件为空"
        else:
            last += "；imencode 编码失败"
    except Exception as e:
        last += f"；imencode 方案异常: {e}"
    return False, last


def list_images(directory):
    """目录下的图片文件名（不含子目录），排序保证可复现。"""
    if not os.path.isdir(directory):
        return []
    return sorted(f for f in os.listdir(directory)
                  if f.lower().endswith(IMG_EXTS) and not f.startswith("."))


def iter_images(root):
    """递归收集目录下的图片绝对路径（跳过隐藏目录），排序保证可复现。

    detect.py 的输入目录常常是 detect_data/ 这类带 real/、fake/ 子目录的结构，
    只看一层会误报“没有可处理的图片”。
    """
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in filenames:
            if name.lower().endswith(IMG_EXTS) and not name.startswith("."):
                found.append(os.path.abspath(os.path.join(dirpath, name)))
    return sorted(found)
