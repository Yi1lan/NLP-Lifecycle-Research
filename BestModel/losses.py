"""Losses and threshold/metric helpers for Stage 3-5."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ThresholdSearchResult:
    threshold: float
    precision: float
    recall: float
    f1: float
    tp: int
    tn: int
    fp: int
    fn: int

    def to_dict(self) -> dict:
        return asdict(self)


def confusion_counts(preds: np.ndarray, labels: np.ndarray) -> tuple[int, int, int, int]:
    preds_i = preds.astype(int)
    labels_i = labels.astype(int)
    tp = int(np.sum((preds_i == 1) & (labels_i == 1)))
    tn = int(np.sum((preds_i == 0) & (labels_i == 0)))
    fp = int(np.sum((preds_i == 1) & (labels_i == 0)))
    fn = int(np.sum((preds_i == 0) & (labels_i == 1)))
    return tp, tn, fp, fn


def precision_recall_f1_from_counts(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def binary_metrics_from_probs(
    probs: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    threshold: float,
) -> ThresholdSearchResult:
    probs_arr = np.asarray(probs, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int64)
    preds = (probs_arr >= threshold).astype(np.int64)
    tp, tn, fp, fn = confusion_counts(preds, labels_arr)
    precision, recall, f1 = precision_recall_f1_from_counts(tp, fp, fn)
    return ThresholdSearchResult(
        threshold=float(threshold),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
    )


def search_best_threshold(
    probs: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    start: float = 0.05,
    end: float = 0.95,
    step: float = 0.005,
) -> ThresholdSearchResult:
    """Grid-search threshold that maximizes positive-class F1."""

    best: ThresholdSearchResult | None = None
    threshold = start
    while threshold <= end + 1e-12:
        current = binary_metrics_from_probs(probs=probs, labels=labels, threshold=threshold)
        if best is None or current.f1 > best.f1:
            best = current
        threshold += step
    if best is None:
        raise RuntimeError("Threshold search failed: no thresholds evaluated.")
    return best


def compute_balanced_class_weights(labels: Sequence[int]) -> torch.Tensor:
    """Compute inverse-frequency class weights for binary labels."""

    labels_arr = np.asarray(labels, dtype=np.int64)
    if labels_arr.size == 0:
        return torch.tensor([1.0, 1.0], dtype=torch.float32)
    n_pos = int(labels_arr.sum())
    n_total = int(labels_arr.size)
    n_neg = n_total - n_pos
    if n_neg == 0 or n_pos == 0:
        return torch.tensor([1.0, 1.0], dtype=torch.float32)
    return torch.tensor(
        [
            n_total / (2.0 * n_neg),
            n_total / (2.0 * n_pos),
        ],
        dtype=torch.float32,
    )


class WeightedFocalLoss(torch.nn.Module):
    """Weighted focal loss for binary (2-class softmax) classification."""

    def __init__(self, class_weights: torch.Tensor | None, gamma: float = 2.0):
        super().__init__()
        self.class_weights = class_weights
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        weights = None
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
        ce = F.cross_entropy(logits, labels, weight=weights, reduction="none")
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()

