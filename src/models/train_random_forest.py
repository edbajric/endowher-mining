from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.data.preprocess_endo import build_endo_preprocessor
from src.data.preprocess_pcos import build_pcos_preprocessor


def train_random_forest(
    x_train,
    y_train,
    dataset_name: str,
    model_out: Path,
    random_state: int = 42,
) -> Pipeline:
    if dataset_name == "pcos":
        preprocessor = build_pcos_preprocessor(x_train, scale_numeric=False)
    elif dataset_name == "endometriosis":
        preprocessor = build_endo_preprocessor(x_train, scale_numeric=False)
    else:
        raise ValueError("dataset_name must be 'pcos' or 'endometriosis'")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=random_state,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_out)
    return pipeline


def train_xgboost_optional(
    x_train,
    y_train,
    dataset_name: str,
    model_out: Path,
    random_state: int = 42,
):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("xgboost is not installed. Install it or skip optional training.") from exc

    if dataset_name == "pcos":
        preprocessor = build_pcos_preprocessor(x_train, scale_numeric=False)
    elif dataset_name == "endometriosis":
        preprocessor = build_endo_preprocessor(x_train, scale_numeric=False)
    else:
        raise ValueError("dataset_name must be 'pcos' or 'endometriosis'")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=4,
                    random_state=random_state,
                    eval_metric="logloss",
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_out)
    return pipeline


def extract_tree_feature_importances(pipeline, feature_names):
    """Return feature importance mapping for tree-based models.

    TODO: Ensure feature_names come from fitted preprocessor.get_feature_names_out().
    """
    model = pipeline.named_steps["model"]
    importance = model.feature_importances_
    return dict(zip(feature_names, importance))
