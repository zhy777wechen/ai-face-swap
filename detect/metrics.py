# -*- coding: utf-8 -*-
"""
评估指标（纯 numpy，不依赖 sklearn）。

只依赖本仓库已有依赖，避免为了几个指标引入新包。
正类约定为 LABEL_FAKE=1，即“判定为伪造”的概率 score 越大越像假。
"""

import numpy as np


def confusion_matrix(labels, preds):
    """返回 (tn, fp, fn, tp)，与 sklearn 的 labels=[0,1] 一致。"""
    labels = np.asarray(labels).astype(int)
    preds = np.asarray(preds).astype(int)
    tp = int(((labels == 1) & (preds == 1)).sum())
    tn = int(((labels == 0) & (preds == 0)).sum())
    fp = int(((labels == 0) & (preds == 1)).sum())
    fn = int(((labels == 1) & (preds == 0)).sum())
    return tn, fp, fn, tp


def classification_report_values(labels, preds):
    """准确率、精确率、召回率(F1)、宏平均 F1；分母为 0 时记 0 而不是报错。"""
    tn, fp, fn, tp = confusion_matrix(labels, preds)
    total = tn + fp + fn + tp
    acc = (tn + tp) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0          # 检出率 ADR
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    p_real = tn / (tn + fn) if (tn + fn) else 0.0
    r_real = tn / (tn + fp) if (tn + fp) else 0.0
    f1_real = 2 * p_real * r_real / (p_real + r_real) if (p_real + r_real) else 0.0
    macro_f1 = (f1 + f1_real) / 2.0
    apcER = fp / (tn + fp) if (tn + fp) else 0.0           # 真实被误判为伪造的比例
    bpcER = fn / (tp + fn) if (tp + fn) else 0.0           # 伪造被漏检的比例
    eer_balanced = (apcER + bpcER) / 2.0
    return {
        "accuracy": acc, "precision_fake": precision, "recall_fake": recall,
        "f1_fake": f1, "macro_f1": macro_f1,
        "APCER": apcER, "BPCER": bpcER, "EER_balanced": eer_balanced,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def auc_roc(labels, scores):
    """ROC AUC，用带并列值的秩统计（Mann-Whitney U），无需绘制曲线。"""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0     # 1-based，并列取平均秩
        i = j + 1
    rank_sum_pos = ranks[labels == 1].sum()
    n1, n0 = pos.size, neg.size
    return float((rank_sum_pos - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def roc_curve(labels, scores, max_points=512):
    """返回 (fpr, tpr, thresholds)，阈值取自去重后的 score 集合。"""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    s = scores[order]
    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)
    P = int((y == 1).sum())
    N = int((y == 0).sum())
    tpr = tps / P if P else np.zeros_like(tps, dtype=np.float64)
    fpr = fps / N if N else np.zeros_like(fps, dtype=np.float64)
    keep = np.ones(len(s), dtype=bool)
    keep[1:] = s[1:] != s[:-1]
    if keep.sum() > max_points:
        idx = np.linspace(0, len(s) - 1, max_points).astype(int)
        keep[:] = False
        keep[idx] = True
    return fpr[keep], tpr[keep], s[keep]


def eer(labels, scores):
    """等错误率及其阈值：在 ROC 上找 |FPR-(1-TPR)| 最小的点。"""
    fpr, tpr, thr = roc_curve(labels, scores)
    if len(fpr) == 0:
        return float("nan"), float("nan")
    diff = np.abs(fpr - (1.0 - tpr))
    i = int(np.argmin(diff))
    return float((fpr[i] + 1.0 - tpr[i]) / 2.0), float(thr[i])


def threshold_sweep(labels, scores, steps=21):
    """给定阈值扫描表，用于报告“阈值—误报率—检出率”的权衡。"""
    rows = []
    for t in np.linspace(0.0, 1.0, steps):
        preds = (np.asarray(scores) >= t).astype(int)
        r = classification_report_values(labels, preds)
        rows.append({"threshold": round(float(t), 3), "accuracy": r["accuracy"],
                     "APCER": r["APCER"], "BPCER": r["BPCER"],
                     "recall_fake": r["recall_fake"], "macro_f1": r["macro_f1"]})
    return rows
