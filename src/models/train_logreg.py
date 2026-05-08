from pathlib import Path
from typing import Dict, Iterable

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data.preprocess_endo import build_endo_preprocessor
from src.data.preprocess_pcos import build_pcos_preprocessor


def train_logistic_regression(
    x_train,
    y_train,
    dataset_name: str,
    model_out: Path,
    random_state: int = 42,
    max_iter: int = 1000,
    class_weight: str = "balanced",
) -> Pipeline:
    if dataset_name == "pcos":
        preprocessor = build_pcos_preprocessor(x_train, scale_numeric=True)
    elif dataset_name == "endometriosis":
        preprocessor = build_endo_preprocessor(x_train, scale_numeric=True)
    else:
        raise ValueError("dataset_name must be 'pcos' or 'endometriosis'")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=max_iter,
                    random_state=random_state,
                    class_weight=class_weight,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_out)
    return pipeline


def extract_logreg_coefficients(
    pipeline: Pipeline, feature_names: Iterable[str]
) -> Dict[str, float]:
    """Return logistic regression coefficients mapped to transformed feature names."""
    # Coefficients are only directly interpretable if binary classification and aligned feature names.
    model = pipeline.named_steps["model"]
    coef = model.coef_[0]
    return dict(zip(feature_names, coef))
