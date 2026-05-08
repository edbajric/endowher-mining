from pathlib import Path

import pandas as pd


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return pd.read_csv(path)


def split_features_target(df: pd.DataFrame, target_column: str):
    if target_column not in df.columns:
        raise KeyError(
            f"Target column '{target_column}' was not found. "
            "Confirm the dataset schema and update config before training."
        )
    x = df.drop(columns=[target_column])
    y = df[target_column]
    return x, y
