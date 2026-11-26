# -*- coding: utf-8 -*-
"""
推理入口：对本地照片/目录给出“真实 / 伪造”判定。

    python detect/detect.py --ckpt detect_runs/<tag>/model_best.pt photo.jpg
    python detect/detect.py --ckpt ... 图片目录 --out detect_infer --csv result.csv

流程与训练/评估严格一致：人脸定位（dlib 优先，Haar 兜底）→ 对齐裁剪 →
空域归一化 (+ 频域 DCT/FFT) → 网络 → “伪造”类概率。
阈值默认取权重里保存的验证集最优阈值，可用 --threshold 覆盖。

用途限定：本脚本是课程攻防实验的检测端演示，判定结果只反映本数据集训练出的
检测器能力，不能用于对真人做身份结论或任何实际风控决策。
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from detect import config
    from detect.face_crop import dlib_available, load_and_crop
    from detect.freq_features import frequency_features
    from detect.image_io import imread_unicode, imwrite_unicode, iter_images
    from detect.inference import pick_device
    from detect.models import SPATIAL_MEAN, SPATIAL_STD, load_checkpoint, softmax_prob_fake
else:
    from . import config
    from .face_crop import dlib_available, load_and_crop
    from .freq_features import frequency_features
    from .image_io import imread_unicode, imwrite_unicode, iter_images
    from .inference import pick_device
    from .models import SPATIAL_MEAN, SPATIAL_STD, load_checkpoint, softmax_prob_fake


def collect_inputs(target):
    if os.path.isdir(target):
        return iter_images(target)
    if os.path.isfile(target):
        return [os.path.abspath(target)]
    raise FileNotFoundError(f"找不到输入: {target}")


def annotate(crop, p_fake, thr, detector_note):
    """把结论画在裁剪图上。文字用英文，避免 cv2.putText 无法渲染中文。"""
    out = crop.copy()
    verdict = "FAKE" if p_fake >= thr else "REAL"
    color = (0, 0, 220) if verdict == "FAKE" else (0, 160, 0)
    h = out.shape[0]
    out = cv2.copyMakeBorder(out, 34, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(out, f"{verdict} p_fake={p_fake:.3f} thr={thr:.2f}", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(out, detector_note, (8, h + 52), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (90, 90, 90), 1)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="对单图/目录做伪造判定")
    ap.add_argument("inputs", nargs="+", help="图片或目录")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--out", default=None, help="标注图输出目录（不填则不写图）")
    ap.add_argument("--csv", default="detect_result.csv")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args(argv)

    device = pick_device(args.device)
    model, meta, ckpt = load_checkpoint(args.ckpt, device=device)
    thr = args.threshold if args.threshold is not None else float(ckpt.get("val_threshold", 0.5))

    paths, seen = [], set()
    for t in args.inputs:
        for p in collect_inputs(t):
            if p not in seen:
                seen.add(p)
                paths.append(p)
    if not paths:
        print("✗ 没有可处理的图片")
        return 2

    crops, kept, methods = [], [], []
    for p in paths:
        crop, method, err = load_and_crop(p, size=config.CROP_SIZE)
        if crop is None and err == "no_face":
            # 训练样本本身就是对齐后的人脸裁剪，输入若已经是裁剪图（或 Haar 检不到），
            # 用整图缩放兜底，保持与训练分布一致；真场景照片仍走上面的定位分支。
            im = imread_unicode(p)
            if im is not None:
                crop = cv2.resize(im, (config.CROP_SIZE, config.CROP_SIZE),
                                  interpolation=cv2.INTER_AREA)
                method = "whole"
        if crop is None:
            print(f"  ! 跳过 {os.path.basename(p)}: {err}")
            continue
        crops.append(crop)
        kept.append(p)
        methods.append(method)
    if not crops:
        print("✗ 所有输入都没能定位到人脸")
        return 3
    n_whole = methods.count("whole")
    if n_whole:
        print(f"  注：{n_whole} 张未检到人脸，按整图缩放处理"
              f"（输入若已是人脸裁剪图，这是预期行为）")

    rows = [{"path": p, "label": 0, "group": os.path.splitext(os.path.basename(p))[0],
             "generator": "", "source_image": ""} for p in kept]
    # 复用同一套预处理：把已裁剪好的人脸写成临时文件代价太高，这里直接喂 crop
    probs = predict_from_crops(model, meta, crops, device, args.batch)
    note_summary = "detector=" + ("dlib" if dlib_available() else "haar")

    print("=" * 56)
    print("检测模块 · 推理")
    print("=" * 56)
    print(f"权重 : {os.path.abspath(args.ckpt)}")
    print(f"结构 : {meta.get('arch')}   阈值 : {thr:.3f}   定位 : {note_summary}"
          + (f"（{n_whole} 张整图兜底）" if n_whole else ""))
    print(f"样本 : {len(rows)} 张\n")

    fake_n = 0
    records = []
    for p, pr, m in zip(kept, probs, methods):
        verdict = config.LABEL_CN[config.LABEL_FAKE] if pr >= thr else config.LABEL_CN[config.LABEL_REAL]
        fake_n += int(pr >= thr)
        print("  {:<48} p_fake={:.3f}  {}".format(os.path.basename(p)[:48], pr, verdict))
        records.append({"image": p, "p_fake": round(float(pr), 4),
                        "verdict": verdict, "threshold": round(thr, 3), "detector": m})

    if args.out:
        out_dir = os.path.abspath(args.out)
        os.makedirs(out_dir, exist_ok=True)
        for i, (p, crop, pr, m) in enumerate(zip(kept, crops, probs, methods), start=1):
            # 目录是递归收集的，不同子目录可能同名，补序号避免互相覆盖
            name = f"{i:04d}_{os.path.splitext(os.path.basename(p))[0]}_detect.jpg"
            imwrite_unicode(os.path.join(out_dir, name), annotate(crop, pr, thr, "detector=" + m))
        print(f"\n✓ 标注图: {out_dir}")

    if args.csv:
        with open(args.csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["image", "p_fake", "verdict", "threshold", "detector"])
            w.writeheader()
            w.writerows(records)
        print(f"✓ 结果表: {os.path.abspath(args.csv)}")

    print(f"\n判定为伪造 {fake_n} / {len(kept)}（阈值 {thr:.3f}，"
          f"低于阈值的 {len(kept) - fake_n} 张仍只说明本检测器未察觉，"
          f"不代表内容真实）")
    return 0


def predict_from_crops(model, meta, crops, device, batch_size):
    """对内存中的人脸裁剪做推理，预处理与 FaceForgeryDataset 完全相同。"""
    import torch

    spatial_list, freq_list = [], []
    for crop in crops:
        s = cv2.resize(crop, (meta.get("img_size", config.IMG_SIZE),
                              meta.get("img_size", config.IMG_SIZE)),
                       interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        s = (s - SPATIAL_MEAN) / SPATIAL_STD
        spatial_list.append(np.transpose(s, (2, 0, 1)))
        if meta.get("arch") == "dual":
            freq_list.append(frequency_features(crop, size=meta.get("freq_size", config.FREQ_SIZE)))

    out = []
    with torch.no_grad():
        for i in range(0, len(crops), batch_size):
            sp = torch.from_numpy(np.stack(spatial_list[i:i + batch_size])).to(device)
            if meta.get("arch") == "dual":
                fq = torch.from_numpy(np.stack(freq_list[i:i + batch_size])).to(device)
                logits = model(sp, fq)
            else:
                logits = model(sp)
            out.append(softmax_prob_fake(logits))
    return np.concatenate(out)


if __name__ == "__main__":
    sys.exit(main())
