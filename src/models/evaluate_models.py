from pathlib import Path
from typing import Any

import pandas as pd
from pandas import Series

from src.utils.metrics import compute_classification_metrics
from src.utils.plotting import save_confusion_matrix, save_roc_curve


def evaluate_binary_model(
    model: Any,
    x_test: pd.DataFrame,
    y_test: Series,
    model_name: str,
    dataset_name: str,
    figures_dir: Path,
    tables_dir: Path,
):
    y_pred = model.predict(x_test)

    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(x_test)[:, 1]

    metrics = compute_classification_metrics(y_test, y_pred, y_prob=y_prob)

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    save_confusion_matrix(
        metrics["confusion_matrix"],
        title=f"{dataset_name}: {model_name} Confusion Matrix",
        output_path=figures_dir / f"{dataset_name}_{model_name}_confusion_matrix.png",
    )

    save_roc_curve(
        y_true=y_test,
        y_prob=y_prob,
        title=f"{dataset_name}: {model_name} ROC Curve",
        output_path=figures_dir / f"{dataset_name}_{model_name}_roc_curve.png",
    )

    table = pd.DataFrame(
        [
            {
                "dataset": dataset_name,
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
            }
        ]
    )
    table.to_csv(tables_dir / f"{dataset_name}_{model_name}_metrics.csv", index=False)
    return metrics, table
