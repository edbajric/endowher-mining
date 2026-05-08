"""Endometriosis preprocessing utilities."""

from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.load_data import load_endometriosis, split_features_target
from src.utils.config import CONFIG


def preprocess_endo() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load and clean endometriosis data and return train/test split."""
    df = load_endometriosis().copy()
    df = df.apply(pd.to_numeric, errors="coerce")

    x, y = split_features_target(df, CONFIG.endo_target_column)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=CONFIG.test_size,
        random_state=CONFIG.random_state,
        stratify=y,
    )
    x_train, x_test = _impute_train_test(x_train, x_test)
    return x_train, x_test, y_train, y_test


def _impute_train_test(
    x_train: pd.DataFrame, x_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    numeric_cols = x_train.select_dtypes(include=["number"]).columns
    x_train = x_train.copy()
    x_test = x_test.copy()
    for col in numeric_cols:
        median = x_train[col].median()
        x_train[col] = x_train[col].fillna(median)
        x_test[col] = x_test[col].fillna(median)
    return x_train, x_test


def build_endo_preprocessor(x, scale_numeric: bool = False) -> ColumnTransformer:
    """Build preprocessing for endometriosis dataset."""
    numeric_features = x.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [c for c in x.columns if c not in numeric_features]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
