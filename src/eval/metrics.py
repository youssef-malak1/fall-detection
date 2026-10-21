"""Event-level evaluation metrics and cross-validation helpers."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


@dataclass
class Metrics:
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    f1: float
    accuracy: float

    def as_dict(self) -> dict:
        return asdict(self)


def compute(y_true, y_pred) -> Metrics:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    zero_division = 0.0
    return Metrics(
        tp=int(tp),
        fp=int(fp),
        tn=int(tn),
        fn=int(fn),
        precision=float(precision_score(y_true, y_pred, zero_division=zero_division)),
        recall=float(recall_score(y_true, y_pred, zero_division=zero_division)),
        f1=float(f1_score(y_true, y_pred, zero_division=zero_division)),
        accuracy=float((tp + tn) / max(1, len(y_true))),
    )


def aggregate(fold_metrics: list[Metrics]) -> dict:
    """Mean and std across folds for every metric."""
    keys = ["precision", "recall", "f1", "accuracy"]
    out = {}
    for key in keys:
        values = np.array([getattr(m, key) for m in fold_metrics])
        out[f"{key}_mean"] = float(values.mean())
        out[f"{key}_std"] = float(values.std())
    out["tp"] = int(sum(m.tp for m in fold_metrics))
    out["fp"] = int(sum(m.fp for m in fold_metrics))
    out["tn"] = int(sum(m.tn for m in fold_metrics))
    out["fn"] = int(sum(m.fn for m in fold_metrics))
    return out
