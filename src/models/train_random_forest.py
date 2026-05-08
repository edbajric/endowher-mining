"""Training helpers for tree-based models."""

from pathlib import Path
from typing import Dict, Iterable, Optional

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.data.preprocess_endo import build_endo_preprocessor
from src.data.preprocess_pcos import build_pcos_preprocessor
from src.utils.io import save_model


def _select_preprocessor(dataset_name: str, x_train, scale_numeric: bool = False):
    if dataset_name == "pcos":
        return build_pcos_preprocessor(x_train, scale_numeric=scale_numeric)
    if dataset_name == "endometriosis":
        return build_endo_preprocessor(x_train, scale_numeric=scale_numeric)
    raise ValueError("dataset_name must be 'pcos' or 'endometriosis'")


def train_random_forest(
    x_train,
    y_train,
    dataset_name: str,
    model_out: Path,
    random_state: int = 42,
    n_estimators: int = 300,
    class_weight: str = "balanced",
) -> Pipeline:
    preprocessor = _select_preprocessor(dataset_name, x_train, scale_numeric=False)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=random_state,
                    class_weight=class_weight,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    save_model(pipeline, model_out)
    return pipeline


def train_decision_tree(
    x_train,
    y_train,
    dataset_name: str,
    model_out: Path,
    random_state: int = 42,
    max_depth: Optional[int] = None,
) -> Pipeline:
    preprocessor = _select_preprocessor(dataset_name, x_train, scale_numeric=False)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", DecisionTreeClassifier(random_state=random_state, max_depth=max_depth)),
        ]
    )
    pipeline.fit(x_train, y_train)
    save_model(pipeline, model_out)
    return pipeline


def train_gradient_boosting(
    x_train,
    y_train,
    dataset_name: str,
    model_out: Path,
    random_state: int = 42,
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    max_depth: int = 3,
) -> Pipeline:
    preprocessor = _select_preprocessor(dataset_name, x_train, scale_numeric=False)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    save_model(pipeline, model_out)
    return pipeline


def train_xgboost_optional(
    x_train,
    y_train,
    dataset_name: str,
    model_out: Path,
    random_state: int = 42,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 4,
):
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        raise RuntimeError(f"XGBoost import failed: {exc}") from exc

    preprocessor = _select_preprocessor(dataset_name, x_train, scale_numeric=False)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                XGBClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    random_state=random_state,
                    eval_metric="logloss",
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    save_model(pipeline, model_out)
    return pipeline


def extract_tree_feature_importances(
    pipeline: Pipeline, feature_names: Iterable[str]
) -> Dict[str, float]:
    """Return feature importance mapping for tree-based models.

    TODO: Ensure feature_names come from fitted preprocessor.get_feature_names_out().
    """
    model = pipeline.named_steps["model"]
    importance = model.feature_importances_
    return dict(zip(feature_names, importance))
