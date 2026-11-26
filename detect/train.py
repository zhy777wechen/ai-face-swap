# -*- coding: utf-8 -*-
"""
训练入口：活体/伪造二分类检测器。

    python detect/train.py --data detect_data --arch dual --epochs 30
    python detect/train.py --data detect_data --arch liveness --epochs 30

要点（与 PPT 第 13 页“训练活体检测 CNN，构建空域加频域双分支检测网络”对应）：
  * --arch liveness 复刻 PPT 第 10 页的浅层 CNN；--arch dual 为空域+频域双分支。
  * 切分按身份分组（dataset.group_split），避免同一底图的真/假样本分居两侧。
  * 默认按少数类下采样平衡类别（--no-balance 关闭），否则 ACC 会被多数类撑高。
  * 每个 epoch 在验证集上记录 loss / ACC / F1 / AUC / EER，按 AUC 选最优权重。
  * 权重内保存 model_meta 与验证集最优阈值，evaluate.py / detect.py 直接复用。

算力受限时（PPT 风险应对）：--epochs 调小、--img-size 32 即可在 CPU 上跑通全流程；
云端 GPU 只需把 --device cuda 传进来，其余参数不变。
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from detect import config, dataset as ds, metrics
    from detect.inference import make_loader, pick_device
    from detect.models import build_model, count_parameters, save_checkpoint
    from detect.models import softmax_prob_fake
else:
    from . import config, dataset as ds, metrics
    from .inference import make_loader, pick_device
    from .models import (build_model, count_parameters, save_checkpoint,
                         softmax_prob_fake)


def forward_one(model, arch, spatial, freq):
    return model(spatial, freq) if arch == "dual" else model(spatial)


@torch.no_grad()
def evaluate_split(model, meta, rows, device, batch_size, criterion):
    """返回 (loss, p_fake, labels)。meta 决定预处理尺寸，与权重保持一致。"""
    model.eval()
    loader = make_loader(rows, meta, batch_size=batch_size)
    total_loss, n = 0.0, 0
    probs, labels = [], []
    for b in loader:
        spatial = b["spatial"].to(device)
        freq = b["freq"].to(device) if b["freq"] is not None else None
        logits = forward_one(model, meta["arch"], spatial, freq)
        loss = criterion(logits, b["label"])
        total_loss += float(loss) * b["label"].numel()
        n += b["label"].numel()
        probs.append(softmax_prob_fake(logits))
        labels.append(b["label"].numpy())
    return total_loss / max(n, 1), np.concatenate(probs), np.concatenate(labels)


def train_one_epoch(model, meta, rows, device, batch_size, criterion, optimizer, epoch):
    model.train()
    loader = make_loader(rows, meta, batch_size=batch_size, augment=True)
    loader.dataset.set_rng(epoch)
    total_loss, n, correct = 0.0, 0, 0
    for b in loader:
        spatial = b["spatial"].to(device)
        freq = b["freq"].to(device) if b["freq"] is not None else None
        y = b["label"].to(device)
        logits = forward_one(model, meta["arch"], spatial, freq)
        loss = criterion(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * y.numel()
        n += y.numel()
        correct += int((logits.argmax(1) == y).sum())
    return total_loss / max(n, 1), correct / max(n, 1)


def best_threshold(labels, probs):
    """在验证集上取 EER 最小处的阈值（与 0.5 相比更贴合实际取舍）。"""
    eer, thr = metrics.eer(labels, probs)
    return thr if np.isfinite(thr) else 0.5


def write_history(path, history):
    import csv
    with open(path, "w", encoding="utf-8", newline="") as f:
        cols = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc",
                "val_macro_f1", "val_auc", "val_eer", "lr", "seconds"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in history:
            w.writerow({c: row.get(c, "") for c in cols})


def main(argv=None):
    ap = argparse.ArgumentParser(description="训练深度伪造/活体检测器")
    ap.add_argument("--data", default=config.DATA_DIR, help="数据集目录（含 manifest.csv）")
    ap.add_argument("--arch", choices=["liveness", "dual"], default="dual")
    ap.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    ap.add_argument("--freq-size", type=int, default=config.FREQ_SIZE)
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=config.LR)
    ap.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    ap.add_argument("--val-ratio", type=float, default=config.VAL_RATIO)
    ap.add_argument("--test-ratio", type=float, default=config.TEST_RATIO)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--device", default=None, help="cpu / cuda，默认自动选择")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--no-balance", action="store_true", help="不做类别平衡下采样")
    ap.add_argument("--out", default=config.RUN_DIR, help="权重与训练历史输出目录")
    ap.add_argument("--tag", default=None, help="本次实验名，默认 arch+时间戳")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = os.path.abspath(args.data)
    run_dir = os.path.abspath(args.out)
    tag = args.tag or f"{args.arch}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = os.path.join(run_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    device = pick_device(args.device)
    print("=" * 56)
    print("检测模块 · 训练")
    print("=" * 56)
    print(f"数据集   : {data_dir}")
    print(f"网络结构 : {args.arch}   输入 空域{args.img_size} / 频域{args.freq_size}")
    print(f"设备     : {device}")
    print(f"输出     : {out_dir}")

    rows = ds.read_manifest(data_dir)
    s = ds.summarize(rows)
    print(f"样本     : 共 {s['total']}（真 {s['real']} / 假 {s['fake']}），身份组 {s['groups']}")

    train, val, test, split_info = ds.group_split(rows, args.val_ratio, args.test_ratio, args.seed)
    if not args.no_balance:
        train = ds.balance_labels(train, seed=args.seed)
        val = ds.balance_labels(val, seed=args.seed + 1)
    print(f"切分     : train {len(train)} / val {len(val)}"
          + (f" / test {len(test)}" if test else "")
          + f"（验证集身份组 {len(split_info['val_groups'])} 个）")
    steps = max(len(train) // args.batch, 1)
    print(f"每 epoch : {steps} 个 optimizer step")
    if steps * args.epochs < 200:
        print("           提示：样本少时 step 数很低，loss 会长时间停在 ln2≈0.693；"
              "把 --batch 调小（如 4~8）或加大 --epochs 即可正常收敛。")

    model = build_model(args.arch, img_size=args.img_size, freq_size=args.freq_size).to(device)
    meta = {"arch": args.arch, "img_size": args.img_size, "freq_size": args.freq_size,
            "freq_channels": 3, "num_classes": 2, "seed": args.seed}
    print(f"参数量   : {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    history, best_auc, best_loss, best_epoch = [], -1.0, float("inf"), 0
    ckpt_path = os.path.join(out_dir, "model_best.pt")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, meta, train, device, args.batch,
                                          criterion, optimizer, epoch)
        v_loss, vp, vy = evaluate_split(model, meta, val, device, args.batch, criterion)
        rep = metrics.classification_report_values(vy, (vp >= 0.5).astype(int))
        auc = metrics.auc_roc(vy, vp)
        eer, _ = metrics.eer(vy, vp)
        scheduler.step()
        row = {"epoch": epoch, "train_loss": round(tr_loss, 4), "train_acc": round(tr_acc, 4),
               "val_loss": round(v_loss, 4), "val_acc": round(rep["accuracy"], 4),
               "val_macro_f1": round(rep["macro_f1"], 4), "val_auc": round(float(auc), 4),
               "val_eer": round(float(eer), 4),
               "lr": optimizer.param_groups[0]["lr"], "seconds": round(time.time() - t0, 1)}
        history.append(row)
        print("epoch {:>3}  train_loss {:.4f} acc {:.3f} | val_loss {:.4f} acc {:.3f} "
              "f1 {:.3f} auc {:.3f} eer {:.3f}".format(
                  epoch, tr_loss, tr_acc, v_loss, rep["accuracy"], rep["macro_f1"], auc, eer))

        auc_ok = np.isfinite(auc) and auc > best_auc + 1e-12
        loss_ok = np.isfinite(auc) and abs(auc - best_auc) <= 1e-12 and v_loss < best_loss - 1e-12
        if auc_ok or loss_ok:
            best_auc, best_loss, best_epoch = float(auc), float(v_loss), epoch
            thr = best_threshold(vy, vp)
            save_checkpoint(ckpt_path, model, meta, extra={
                "train_history": history,
                "best_epoch": best_epoch,
                "val_metrics": {**rep, "auc": auc, "eer": eer},
                "val_threshold": thr,
                "data_dir": data_dir,
                "split": split_info,
                "device": str(device),
            })

    write_history(os.path.join(out_dir, "history.csv"), history)

    _, vp_last, vy_last = evaluate_split(model, meta, val, device, args.batch, criterion)
    final_thr = best_threshold(vy_last, vp_last)
    save_checkpoint(os.path.join(out_dir, "model_last.pt"), model, meta,
                    extra={"best_epoch": best_epoch, "val_threshold": final_thr,
                           "data_dir": data_dir, "split": split_info})

    print("\n✓ 训练完成")
    print(f"  最优 epoch : {best_epoch}（val AUC {best_auc:.4f}）")
    print(f"  最优权重   : {ckpt_path}")
    print(f"  训练历史   : {os.path.join(out_dir, 'history.csv')}")
    print(f"\n下一步: python detect/evaluate.py --ckpt {ckpt_path} --data {data_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
