"""IO utilities for the project."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def ensure_dir(path: Path) -> None:
	"""Create directory if it does not exist."""
	path.mkdir(parents=True, exist_ok=True)


def save_dataframe(df: pd.DataFrame, path: Path, index: bool = False) -> None:
	"""Save a DataFrame to CSV and ensure parent directory exists."""
	ensure_dir(path.parent)
	df.to_csv(path, index=index)


def save_model(model: Any, path: Path) -> None:
	"""Persist a model with joblib and ensure parent directory exists."""
	ensure_dir(path.parent)
	joblib.dump(model, path)
