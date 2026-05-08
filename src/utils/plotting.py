from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import RocCurveDisplay


def save_confusion_matrix(cm, title: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_roc_curve(y_true, y_prob, title: str, output_path: Path) -> None:
    if y_prob is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true=y_true, y_score=y_prob, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_roc_curves(
    y_true,
    model_probabilities: Dict[str, Optional[object]],
    title: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, y_prob in model_probabilities.items():
        if y_prob is None:
            continue
        RocCurveDisplay.from_predictions(y_true=y_true, y_score=y_prob, ax=ax, name=name)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_feature_importances(
    features: Iterable[str],
    importances: Iterable[float],
    title: str,
    output_path: Path,
    top_n: int = 25,
) -> None:
    data = list(zip(features, importances))
    data.sort(key=lambda item: item[1], reverse=True)
    data = data[:top_n]
    labels, values = zip(*data) if data else ([], [])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))
    sns.barplot(x=list(values), y=list(labels), orient="h", color="C0", legend=False)
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
