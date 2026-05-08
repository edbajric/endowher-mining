"""PCOS preprocessing utilities."""

from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.load_data import load_pcos, split_features_target
from src.utils.config import CONFIG

DROP_COLS = ["Sl. No", "Patient File No.", "Unnamed: 44"]
BINARY_YN_COLS = [
    "Weight gain(Y/N)",
    "hair growth(Y/N)",
    "Skin darkening (Y/N)",
    "Hair loss(Y/N)",
    "Pimples(Y/N)",
    "Fast food (Y/N)",
    "Reg.Exercise(Y/N)",
    "Pregnant(Y/N)",
]


def _coerce_numeric(df: pd.DataFrame, exclude: List[str]) -> pd.DataFrame:
    for col in df.columns:
        if col in exclude:
            continue
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _clean_binary_columns(df: pd.DataFrame, binary_cols: List[str]) -> pd.DataFrame:
    mapping = {"Y": 1, "N": 0, "Yes": 1, "No": 0, "y": 1, "n": 0}
    for col in binary_cols:
        if col not in df.columns:
            continue
        df[col] = df[col].replace(mapping)
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def preprocess_pcos() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load and clean PCOS data and return train/test split."""
    df = load_pcos().copy()
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    df = _clean_binary_columns(df, BINARY_YN_COLS)
    df = _coerce_numeric(df, exclude=[CONFIG.pcos_target_column])

    x, y = split_features_target(df, CONFIG.pcos_target_column)

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
    categorical_cols = [c for c in x_train.columns if c not in numeric_cols]

    x_train = x_train.copy()
    x_test = x_test.copy()

    for col in numeric_cols:
        median = x_train[col].median()
        x_train[col] = x_train[col].fillna(median)
        x_test[col] = x_test[col].fillna(median)

    for col in categorical_cols:
        mode = x_train[col].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else ""
        x_train[col] = x_train[col].fillna(fill_value)
        x_test[col] = x_test[col].fillna(fill_value)

    return x_train, x_test


def build_pcos_preprocessor(x, scale_numeric: bool = False) -> ColumnTransformer:
    """Build preprocessing pipeline for PCOS features."""
    numeric_features = x.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [c for c in x.columns if c not in numeric_features]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(steps=numeric_steps)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
