# -*- coding: utf-8 -*-
"""
自建数据集生成器（对应 PPT 风险应对“先自建数据集跑通全流程”）。

两种生成方式：

1. --generator swap（默认，产出可用于结论的样本）
   真样本：--gallery 目录里每张原始照片的人脸裁剪。
   假样本：对 gallery 中所有 (底图, 源脸) 有序对，复用 src/face_swap_fixed.py
   的换脸流水线合成，再裁剪人脸。
   源文件零改动：本脚本用 importlib 把 src/face_swap_fixed.py 当模块加载，
   仅补上它 main() 里才初始化的 predictor 全局量，算法步骤完全照搬其实现，
   因此这里的假样本与手工运行 `python src/face_swap_fixed.py` 的结果一致。
   需要 dlib 与 shape_predictor_68_face_landmarks.dat（见 README）。

2. --generator synthetic（无 dlib / 无素材时的流水线自检）
   程序化生成“人脸状”图像：真样本为平滑合成人脸，假样本按几何换脸的
   痕迹合成方式做区域贴回 + 重采样 + 羽化融合，保留插值与频域伪影。
   ⚠ 只用于验证训练/评估链路能跑通，其指标不能作为研究结论。

用法：
    python detect/build_dataset.py --gallery 素材目录 --out detect_data
    python detect/build_dataset.py --generator synthetic --samples 120 --out detect_data
"""

import argparse
import importlib.util
import os
import sys

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from detect import config, manifest as mf
    from detect.face_crop import dlib_available, load_and_crop
    from detect.image_io import imwrite_unicode, list_images
else:
    from . import config
    from . import manifest as mf
    from .face_crop import dlib_available, load_and_crop
    from .image_io import imwrite_unicode, list_images

SWAP_SCRIPT = os.path.join(config.SRC_DIR, "face_swap_fixed.py")


def _load_swap_module():
    """把 src/face_swap_fixed.py 当模块导入，并补齐它 main() 里初始化的全局量。

    该模块级代码会 os.chdir 到 src/（为了让 dlib 用相对名打开 .dat），
    所以导入后要恢复调用方的工作目录。
    """
    if not os.path.exists(SWAP_SCRIPT):
        raise FileNotFoundError(f"找不到攻击侧脚本: {SWAP_SCRIPT}")
    cwd = os.getcwd()
    try:
        spec = importlib.util.spec_from_file_location("faceswap_baseline", SWAP_SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # face_swap_fixed.main() 里才给全局 predictor 赋值，这里外部补上
        mod.predictor = mod.dlib.shape_predictor(mod.DAT_NAME)
    finally:
        os.chdir(cwd)
    if not os.path.exists(os.path.join(config.SRC_DIR, mod.DAT_NAME)):
        raise FileNotFoundError(f"缺少 {mod.DAT_NAME}，请放到 src/ 目录（README 有下载地址）")
    return mod


def swap_face(mod, base_path, source_path):
    """照搬 face_swap_fixed.main() 的合成步骤，返回 BGR uint8 结果。"""
    im1, lm1 = mod.read_im_and_landmarks(base_path)
    im2, lm2 = mod.read_im_and_landmarks(source_path)
    M = mod.transformation_from_points(lm1[mod.ALIGN_POINTS], lm2[mod.ALIGN_POINTS])
    mask = mod.get_face_mask(im2, lm2)
    warped_mask = mod.warp_im(mask, M, im1.shape)
    combined = np.max([mod.get_face_mask(im1, lm1), warped_mask], axis=0)
    cover = mod.warp_im(np.full(im2.shape, 255, np.uint8), M, im1.shape)[:, :, 0] > 0
    combined = combined * cover[:, :, None]
    warped2 = mod.warp_im(im2, M, im1.shape)
    corrected = mod.correct_colours(im1, warped2, lm1)
    out = im1 * (1.0 - combined) + corrected * combined
    return np.clip(out, 0, 255).astype(np.uint8)


def build_from_swap(gallery_dir, out_dir, crop_size, log):
    """真样本 = 原图人脸裁剪；假样本 = src 换脸结果的人脸裁剪。"""
    images = [os.path.join(gallery_dir, f) for f in list_images(gallery_dir)]
    if len(images) < 2:
        raise ValueError(f"--gallery 目录下至少需要 2 张照片，当前 {len(images)} 张")

    mod = _load_swap_module()
    real_dir = os.path.join(out_dir, "real")
    fake_dir = os.path.join(out_dir, "fake")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    rows = []
    crops = {}
    tmp = os.path.join(out_dir, "_swap_tmp.jpg")
    for p in images:
        stem = os.path.splitext(os.path.basename(p))[0]
        crop, method, err = load_and_crop(p, size=crop_size)
        if crop is None:
            log(f"  ! 跳过 {os.path.basename(p)}: {err}")
            continue
        rel = os.path.join("real", f"{stem}.jpg")
        ok, info = imwrite_unicode(os.path.join(out_dir, rel), crop)
        if not ok:
            log(f"  ! 写入失败 {info}")
            continue
        crops[stem] = p
        rows.append({"path": rel.replace(os.sep, "/"), "label": config.LABEL_REAL,
                     "group": stem, "generator": f"swap:{method}",
                     "source_image": os.path.basename(p)})
        log(f"  真实 {rel}  定位={method}")

    for base_stem, base_path in crops.items():
        for src_stem, src_path in crops.items():
            if base_stem == src_stem:
                continue
            try:
                merged = swap_face(mod, base_path, src_path)
            except Exception as e:
                log(f"  ! 换脸失败 {base_stem}<-{src_stem}: {e}")
                continue
            if not imwrite_unicode(tmp, merged)[0]:
                log(f"  ! 中间结果写入失败: {tmp}")
                continue
            crop, method, err = load_and_crop(tmp, size=crop_size)
            if crop is None:
                log(f"  ! 伪造样本无脸 {base_stem}<-{src_stem}: {err}")
                continue
            name = f"{base_stem}__from__{src_stem}.jpg"
            rel = os.path.join("fake", name)
            ok, _ = imwrite_unicode(os.path.join(out_dir, rel), crop)
            if not ok:
                continue
            # group 用底图身份：真实底图与以它为底的伪造样本必须同侧，防止身份泄漏
            rows.append({"path": rel.replace(os.sep, "/"), "label": config.LABEL_FAKE,
                         "group": base_stem, "generator": f"swap:{method}",
                         "source_image": f"{os.path.basename(base_path)}+{os.path.basename(src_path)}"})
            log(f"  伪造 {rel}")
    if os.path.exists(tmp):
        os.remove(tmp)
    return rows


# ---------------------------------------------------------------- 合成素材

def _synth_identity(rng, size):
    """画一张低频“人脸状”图：椭圆底色 + 光照梯度 + 眉眼嘴位置的大致结构。"""
    img = np.zeros((size, size), np.float32)
    yy, xx = np.ogrid[:size, :size]
    cy, cx = size * 0.52, size * 0.5
    rx, ry = size * (0.24 + rng.random() * 0.06), size * (0.32 + rng.random() * 0.06)
    face = np.exp(-(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2))
    img += face * (0.55 + rng.random() * 0.25)

    light = np.linspace(0.75, 1.15, size, dtype=np.float32)[None, :]
    img *= light

    def blob(x, y, r, amp):
        m = np.exp(-(((xx - x) ** 2 + (yy - y) ** 2) / (2 * (size * r) ** 2)))
        img[:, :] -= m * amp

    eye_y = size * 0.42
    eye_dx = size * (0.11 + rng.random() * 0.04)
    blob(cx - eye_dx, eye_y, 0.022, 0.35)
    blob(cx + eye_dx, eye_y, 0.022, 0.35)
    blob(cx - eye_dx, eye_y - size * 0.06, 0.045, 0.10)
    blob(cx + eye_dx, eye_y - size * 0.06, 0.045, 0.10)
    blob(cx, size * 0.56, 0.028, 0.14)
    blob(cx, size * 0.70, 0.055, 0.18)
    img += rng.standard_normal(img.shape).astype(np.float32) * 0.012

    # 单通道亮度图按随机色调拉伸成 BGR，得到不同“身份”的肤色差异
    gray = np.clip(img, 0, 1).astype(np.float32)
    tint = (0.85 + rng.random(3) * 0.30).astype(np.float32)
    hsv = gray[:, :, None] * tint[None, None, :]
    return np.clip(hsv * 255, 0, 255).astype(np.uint8)


def _synth_fake_pair(base, src, rng, size):
    """模拟几何换脸的痕迹：区域贴回 + 色彩迁移 + 重采样 + 羽化融合。"""
    h, w = base.shape[:2]
    warped = cv2.resize(src, (int(w * 1.15), int(h * 1.15)))
    warped = cv2.warpAffine(warped, np.float32([[1, 0, -w * 0.07], [0, 1, -h * 0.05]]), (w, h))
    warped = cv2.resize(cv2.resize(warped, (int(w * 1.3), int(h * 1.3)),
                                   interpolation=cv2.INTER_CUBIC), (w, h),
                        interpolation=cv2.INTER_AREA)      # 上/下采样留下插值痕迹

    mask = np.zeros((h, w), np.float32)
    cv2.ellipse(mask, (w // 2, int(h * 0.55)), (int(w * 0.26), int(h * 0.33)),
                0, 0, 360, 1.0, -1)
    k = int(size * 0.09) | 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)

    base_f, war_f = base.astype(np.float32), warped.astype(np.float32)
    b_mean = base_f[mask > 0.5].mean((0, 1)) if (mask > 0.5).any() else base_f.mean((0, 1))
    w_mean = war_f[mask > 0.5].mean((0, 1)) if (mask > 0.5).any() else war_f.mean((0, 1))
    corrected = np.clip(war_f * (b_mean / np.maximum(w_mean, 1.0)), 0, 255)
    out = base_f * (1 - mask[:, :, None]) + corrected * mask[:, :, None]
    jitter = rng.normal(0, 0.6, out.shape).astype(np.float32)   # 融合区轻微噪声不匹配
    out = np.clip(out + jitter * mask[:, :, None], 0, 255).astype(np.uint8)
    return cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA)


def build_synthetic(out_dir, samples, crop_size, seed, log):
    """程序化合成数据集：每个身份 1 个真样本 + 若干个由它衍生的假样本。"""
    rng = np.random.default_rng(seed)
    n_ids = max(4, samples // 3)
    real_dir = os.path.join(out_dir, "real")
    fake_dir = os.path.join(out_dir, "fake")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    rows, ids = [], []
    for i in range(n_ids):
        face = cv2.resize(_synth_identity(np.random.default_rng(seed + i), crop_size),
                          (crop_size, crop_size))
        ids.append(face)
        rel = os.path.join("real", f"id{i:03d}.jpg")
        imwrite_unicode(os.path.join(out_dir, rel), face)
        rows.append({"path": rel.replace(os.sep, "/"), "label": config.LABEL_REAL,
                     "group": f"id{i:03d}", "generator": "synthetic", "source_image": f"id{i:03d}"})

    for i, base in enumerate(ids):
        for j, src in enumerate(ids):
            if i == j or len(rows) >= samples * 2:
                continue
            fake = _synth_fake_pair(base, src, rng, crop_size)
            rel = os.path.join("fake", f"id{i:03d}__from__id{j:03d}.jpg")
            imwrite_unicode(os.path.join(out_dir, rel), fake)
            rows.append({"path": rel.replace(os.sep, "/"), "label": config.LABEL_FAKE,
                         "group": f"id{i:03d}", "generator": "synthetic",
                         "source_image": f"id{i:03d}+id{j:03d}"})
        if len(rows) >= samples * 2:
            break
    log(f"  合成样本 {len(rows)} 条（身份组 {n_ids} 个）")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="自建 real/fake 人脸数据集生成器")
    ap.add_argument("--gallery", help="原始照片目录（swap 模式必填）")
    ap.add_argument("--out", default=config.DATA_DIR, help="数据集输出目录")
    ap.add_argument("--generator", choices=["swap", "synthetic"], default="swap",
                    help="swap=复用 src 换脸流水线；synthetic=流水线自检用")
    ap.add_argument("--crop-size", type=int, default=config.CROP_SIZE)
    ap.add_argument("--samples", type=int, default=120, help="synthetic 模式的样本规模上限")
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args(argv)

    def log(*a):
        print(*a)

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    log("=" * 56)
    log("检测模块 · 自建数据集生成")
    log("=" * 56)
    log(f"输出目录 : {out_dir}")
    log(f"生成方式 : {args.generator}")

    if args.generator == "swap":
        if not args.gallery:
            log("✗ swap 模式需要 --gallery 指定照片目录")
            return 1
        if not dlib_available():
            log("✗ 未检测到可用的 dlib 或 shape_predictor_68_face_landmarks.dat")
            log("  请先 pip install dlib-bin 并把模型放到 src/，或改用 --generator synthetic 自检")
            return 2
        rows = build_from_swap(os.path.abspath(args.gallery), out_dir, args.crop_size, log)
    else:
        rows = build_synthetic(out_dir, args.samples, args.crop_size, args.seed, log)

    if not rows:
        log("✗ 没有产出任何样本")
        return 3
    path = mf.write_manifest(out_dir, rows)
    s = mf.summarize(rows)
    log("\n✓ manifest 写入 : {}".format(path))
    log("  样本总数 : {}（真实 {} / 伪造 {}），身份组 {}".format(
        s["total"], s["real"], s["fake"], s["groups"]))
    log("  定位方式 : {}".format(s["generators"]))
    if args.generator == "synthetic":
        log("\n⚠ synthetic 样本仅用于验证训练/评估链路，指标不可作为研究结论")
    log("\n下一步: python detect/train.py --data {}".format(out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
