# -*- coding: utf-8 -*-
"""
人脸定位与对齐裁剪。

检测器按优先级自动选择，并把用的哪一种写进 manifest 的 generator 列，
这样评估时可以按定位方式分组看指标：
  1. dlib HOG + 68 点模型（与 src 攻击侧同一套，点集对齐最稳）
  2. OpenCV Haar 级联（无 dlib 或模型缺失时的兜底，只做框裁剪、不做旋转对齐）

对齐做法：取左右眼中心连线，用相似变换把双眼放到裁剪框的既定位置，
消除姿态与尺度差异，避免检测器本身的尺度变化被网络当成“伪造线索”。
"""

import os

import cv2
import numpy as np

from . import config
from .image_io import imread_unicode

DAT_NAME = "shape_predictor_68_face_landmarks.dat"
RIGHT_EYE = slice(36, 42)
LEFT_EYE = slice(42, 48)

_dlib_state = {"detector": None, "predictor": None, "tried": False}
_haar_state = {"cascade": None}


def _try_load_dlib():
    """加载 dlib 检测器与 68 点模型；任何一步失败都返回 None 而不抛异常。

    dlib 用窄字符 fopen，含中文的绝对路径会 RuntimeError，所以临时把 CWD
    切到 src/（模型约定放在那里）后用相对文件名打开，随后恢复 CWD。
    """
    if _dlib_state["tried"]:
        return _dlib_state["detector"], _dlib_state["predictor"]
    _dlib_state["tried"] = True
    try:
        import dlib
    except ImportError:
        return None, None

    candidates = [os.path.join(config.SRC_DIR, DAT_NAME), os.path.join(config.MODULE_DIR, DAT_NAME)]
    dat = next((p for p in candidates if os.path.exists(p)), None)
    if dat is None:
        return None, None

    cwd = os.getcwd()
    try:
        os.chdir(os.path.dirname(dat))
        _dlib_state["detector"] = dlib.get_frontal_face_detector()
        _dlib_state["predictor"] = dlib.shape_predictor(os.path.basename(dat))
    except Exception:
        _dlib_state["detector"] = _dlib_state["predictor"] = None
    finally:
        os.chdir(cwd)
    return _dlib_state["detector"], _dlib_state["predictor"]


def _haar():
    if _haar_state["cascade"] is None:
        xml = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        _haar_state["cascade"] = cv2.CascadeClassifier(xml)
    return _haar_state["cascade"]


def dlib_available():
    detector, predictor = _try_load_dlib()
    return detector is not None and predictor is not None


def _boxes_from_dlib(im):
    detector, predictor = _try_load_dlib()
    rects = detector(im, 1)
    out = []
    for r in rects:
        box = (max(r.left(), 0), max(r.top(), 0),
               max(r.width(), 1), max(r.height(), 1))
        pts = np.array([[p.x, p.y] for p in predictor(im, r).parts()], dtype=np.float32)
        out.append((box, pts))
    return out


def _boxes_from_haar(im):
    cascade = _haar()
    if cascade.empty():
        return []
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                     minSize=(36, 36))
    return [((int(x), int(y), int(w), int(h)), None) for (x, y, w, h) in faces]


def detect_faces(im):
    """返回 [(box, landmarks68 或 None, 方法名), ...]，box = (x, y, w, h)。"""
    if dlib_available():
        found = _boxes_from_dlib(im)
        if found:
            return [(box, pts, "dlib") for box, pts in found]
        # dlib 没检到脸时退回 Haar，避免整批样本被丢弃
    return [(box, pts, "haar") for box, pts in _boxes_from_haar(im)]


def _expand_box(box, w, h, margin=0.35):
    x, y, bw, bh = box
    ex, ey = bw * margin, bh * margin
    x0, y0 = max(0, int(x - ex)), max(0, int(y - ey))
    x1, y1 = min(w, int(x + bw + ex)), min(h, int(y + bh + ey))
    side = max(1, x1 - x0, y1 - y0)
    return x0, y0, side


def _align_by_eyes(im, pts, box, size):
    """以双眼为基准做相似变换，把脸摆正后裁成 size x size。"""
    reye = pts[RIGHT_EYE].mean(0)
    leye = pts[LEFT_EYE].mean(0)
    eye_dist = float(np.linalg.norm(leye - reye))
    if eye_dist < 1.0:
        return None
    # 目标模板：双眼落在裁剪框 0.31/0.69 宽、0.40 高的位置（对齐数据集常用取值）
    src = np.array([reye, leye], dtype=np.float32)
    dst = np.array([[size * 0.31, size * 0.40], [size * 0.69, size * 0.40]], dtype=np.float32)
    M = cv2.estimateAffinePartial2D(src, dst)[0]
    if M is None:
        return None
    return cv2.warpAffine(im, M, (size, size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def crop_face(im, box, pts, size=config.CROP_SIZE, method="dlib"):
    """把一张脸裁成 size x size；优先用 68 点对齐，否则做带边距的方框裁剪。"""
    if pts is not None and method == "dlib":
        crop = _align_by_eyes(im, pts, box, size)
        if crop is not None:
            return crop
    x0, y0, side = _expand_box(box, im.shape[1], im.shape[0])
    crop = im[y0:y0 + side, x0:x0 + side]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)


def best_face(im, size=config.CROP_SIZE):
    """取面积最大的人脸（课程场景为单人照）。返回 (裁剪图, 方法名) 或 (None, None)。"""
    faces = detect_faces(im)
    if not faces:
        return None, None
    faces.sort(key=lambda f: f[0][2] * f[0][3], reverse=True)
    box, pts, method = faces[0]
    return crop_face(im, box, pts, size=size, method=method), method


def load_and_crop(path, size=config.CROP_SIZE):
    im = imread_unicode(path)
    if im is None:
        return None, None, "decode_failed"
    crop, method = best_face(im, size=size)
    if crop is None:
        return None, None, "no_face"
    return crop, method, None
