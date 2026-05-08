"""Metrics helpers for classification."""

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true,
    y_pred,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Compute standard binary classification metrics.

    Returns a dict with accuracy, precision, recall, f1, roc_auc, and confusion_matrix.
    """
    metrics: Dict[str, object] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["roc_auc"] = np.nan
    else:
        metrics["roc_auc"] = np.nan

    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)
    return metrics
