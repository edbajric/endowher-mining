"""Dataset loading helpers."""

from pathlib import Path
from typing import Tuple

import pandas as pd

from src.utils.config import CONFIG


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file with validation."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return pd.read_csv(path)


def load_pcos() -> pd.DataFrame:
    """Load the PCOS dataset from Excel."""
    if not CONFIG.pcos_no_infertility_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {CONFIG.pcos_no_infertility_path}"
        )
    return pd.read_excel(CONFIG.pcos_no_infertility_path, sheet_name="Full_new")


def load_endometriosis() -> pd.DataFrame:
    """Load the endometriosis dataset from CSV."""
    return load_csv(CONFIG.endo_raw_path)


def split_features_target(
    df: pd.DataFrame, target_column: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into features and target."""
    if target_column not in df.columns:
        raise KeyError(
            f"Target column '{target_column}' was not found. "
            "Confirm the dataset schema and update config before training."
        )
    x = df.drop(columns=[target_column])
    y = df[target_column]
    return x, y
