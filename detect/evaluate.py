# -*- coding: utf-8 -*-
"""
评估入口：在留出集上量化检测器性能（PPT 第 13 页“在自建与公开数据集上评估”）。

    python detect/evaluate.py --ckpt detect_runs/<tag>/model_best.pt --data detect_data
    python detect/evaluate.py --ckpt ... --split all --threshold 0.5

输出（默认写到 detect_runs/eval_<时间戳>/）：
    report.csv        总体指标 + 分组指标（按定位方式/生成方式、按底图身份）
    sweep.csv         阈值—ACC—APCER—BPCER 权衡表
    errors.csv        判错样本清单（供人工看融合边界等线索）
    roc.csv           ROC 曲线点，便于画图
控制台同步打印可读摘要。
"""

import argparse
import csv
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from detect import config, dataset as ds, metrics
    from detect.inference import pick_device, predict_rows
    from detect.models import load_checkpoint
else:
    from . import config, dataset as ds, metrics
    from .inference import pick_device, predict_rows
    from .models import load_checkpoint


def fmt(rep):
    return ("ACC {:.4f}  F1(假) {:.4f}  召回 {:.4f}  AUC {}  EER {}  "
            "APCER {:.4f}  BPCER {:.4f}").format(
        rep["accuracy"], rep["f1_fake"], rep["recall_fake"], rep["_auc"],
        rep["_eer"], rep["APCER"], rep["BPCER"])


def with_auc_eer(labels, probs, preds):
    rep = metrics.classification_report_values(labels, preds)
    rep["_auc"] = round(float(metrics.auc_roc(labels, probs)), 4)
    rep["_eer"], rep["_eer_thr"] = [round(float(v), 4) if np.isfinite(v) else v
                                    for v in metrics.eer(labels, probs)]
    return rep


def group_breakdown(rows, labels, probs, preds):
    """按 generator 与按底图身份分组给指标，用来发现“只靠某类线索”的过拟合。"""
    out = []
    for key, name in (("generator", "generator"), ("source_image", "source_image")):
        buckets = defaultdict(list)
        for r, y, p, pr in zip(rows, labels, probs, preds):
            buckets[r.get(key, "") or "-"].append((y, p, pr))
        for g, items in sorted(buckets.items()):
            if len(items) < 2 or len({y for y, _, _ in items}) < 2:
                continue
            yy = np.array([i[0] for i in items])
            pp = np.array([i[1] for i in items])
            pr = np.array([i[2] for i in items])
            out.append({"level": name, "group": g, "n": len(items),
                        **with_auc_eer(yy, pp, pr)})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="评估伪造检测器")
    ap.add_argument("--ckpt", required=True, help="train.py 产出的 .pt")
    ap.add_argument("--data", default=config.DATA_DIR)
    ap.add_argument("--split", choices=["val", "test", "all"], default="val")
    ap.add_argument("--threshold", type=float, default=None,
                    help="判定阈值；默认用权重里存的验证集最优阈值")
    ap.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    data_dir = os.path.abspath(args.data)
    rows = ds.read_manifest(data_dir)
    train, val, test, split_info = ds.group_split(rows, config.VAL_RATIO, config.TEST_RATIO)

    if args.split == "all":
        eval_rows = rows
    elif args.split == "test":
        if not test:
            print("✗ 数据集切分时未留 test 组，请用 --split val，或训练时加 --test-ratio 0.2")
            return 2
        eval_rows = test
    else:
        eval_rows = val

    device = pick_device(args.device)
    model, meta, ckpt = load_checkpoint(args.ckpt, device=device)
    thr = args.threshold
    if thr is None:
        thr = float(ckpt.get("val_threshold", 0.5))

    print("=" * 56)
    print("检测模块 · 评估")
    print("=" * 56)
    print(f"权重     : {os.path.abspath(args.ckpt)}")
    print(f"结构     : {meta.get('arch')}  输入 空域{meta.get('img_size')} / 频域{meta.get('freq_size')}")
    print(f"训练信息 : best_epoch={ckpt.get('best_epoch', '-')}  "
          f"data_dir={ckpt.get('data_dir', '-')}")
    print(f"评估集   : {args.split} 共 {len(eval_rows)} 条   阈值={thr:.3f}   设备={device}")

    probs, labels = predict_rows(model, meta, eval_rows, device, args.batch)
    preds = (probs >= thr).astype(int)
    overall = with_auc_eer(labels, probs, preds)
    print("\n总体指标")
    print("  " + fmt(overall))
    tn, fp, fn, tp = overall["tn"], overall["fp"], overall["fn"], overall["tp"]
    print(f"  混淆矩阵  TN={tn} FP={fp}  FN={fn} TP={tp}")

    per_gen = Counter(r["generator"] for r in eval_rows)
    print(f"  定位/生成方式分布: {dict(per_gen)}")

    groups = group_breakdown(eval_rows, labels, probs, preds)
    if groups:
        print("\n分组指标（组内同时含真伪样本才有意义）")
        for g in groups:
            print("  [{}] {:<28} n={:<4} {}".format(
                g["level"], (g["group"] or "-")[:28], g["n"], fmt(g)))

    sweep = metrics.threshold_sweep(labels, probs)

    out_dir = os.path.abspath(args.out or os.path.join(
        config.RUN_DIR, f"eval_{time.strftime('%Y%m%d_%H%M%S')}"))
    os.makedirs(out_dir, exist_ok=True)

    flat_overall = {"level": "overall", "group": args.split, "n": len(eval_rows), **overall}
    cols = ["level", "group", "n", "accuracy", "precision_fake", "recall_fake",
            "f1_fake", "macro_f1", "APCER", "BPCER", "_auc", "_eer", "_eer_thr",
            "tn", "fp", "fn", "tp"]
    with open(os.path.join(out_dir, "report.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in [flat_overall] + groups:
            w.writerow(row)

    with open(os.path.join(out_dir, "sweep.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep[0].keys()))
        w.writeheader()
        w.writerows(sweep)

    fpr, tpr, thr_list = metrics.roc_curve(labels, probs)
    with open(os.path.join(out_dir, "roc.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fpr", "tpr", "threshold"])
        w.writerows(zip(fpr, tpr, thr_list))

    err_rows = []
    for r, y, p, pr in zip(eval_rows, labels, probs, preds):
        if y != pr:
            err_rows.append({"path": r["path"], "label": int(y), "p_fake": round(float(p), 4),
                             "pred": int(pr), "kind": "漏检(FN)" if y == 1 else "误报(FP)",
                             "group": r["group"], "generator": r["generator"]})
    err_rows.sort(key=lambda d: -abs(d["p_fake"] - 0.5))
    with open(os.path.join(out_dir, "errors.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "p_fake", "pred", "kind",
                                          "group", "generator"])
        w.writeheader()
        w.writerows(err_rows)

    print(f"\n判错 {len(err_rows)} / {len(eval_rows)} 条")
    for e in err_rows[:10]:
        print("  {:<52} p_fake={:.3f} {}".format(
            os.path.basename(e["path"])[:52], e["p_fake"], e["kind"]))

    print(f"\n✓ 报告输出: {out_dir}")
    print("  report.csv / sweep.csv / roc.csv / errors.csv")
    print("\n提示: 若要进入第 04 阶段对抗实验，用同一份权重做 FGSM/PGD，"
          "对比 sweep.csv 中同阈值下的检出率落差。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
