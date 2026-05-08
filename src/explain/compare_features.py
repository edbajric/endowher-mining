"""Feature importance comparison helpers."""

from pathlib import Path
from typing import List

import pandas as pd

from src.data.feature_mapping import normalize_feature_name


def _assign_conceptual_group(feature_name: str) -> str:
    key = normalize_feature_name(feature_name)
    if any(token in key for token in ["cycle", "menstrual", "period"]):
        return "cycle_menstrual"
    if any(token in key for token in ["hormone", "fsh", "lh", "amh", "tsh", "prl", "prog"]):
        return "hormonal_indicators"
    if any(token in key for token in ["pain", "chronic_pain"]):
        return "pain"
    if any(token in key for token in ["infertility", "pregnant", "aborption", "fertility"]):
        return "infertility_fertility"
    if any(token in key for token in ["bmi", "weight", "waist", "hip", "body"]):
        return "bmi_body_composition"
    return "other"


def _prepare_table(df: pd.DataFrame, dataset_label: str) -> pd.DataFrame:
    if "feature" not in df.columns or "importance" not in df.columns:
        raise KeyError(
            "Feature importance tables must include 'feature' and 'importance' columns."
        )
    df = df.copy()
    df["dataset"] = dataset_label
    df["conceptual_group"] = df["feature"].map(_assign_conceptual_group)
    return df[["feature", "dataset", "importance", "conceptual_group"]]


def compare_feature_importance_tables(
    pcos_table_path: Path,
    endo_table_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Create a combined feature comparison table with conceptual groups."""
    pcos = pd.read_csv(pcos_table_path)
    endo = pd.read_csv(endo_table_path)

    combined = pd.concat(
        [_prepare_table(pcos, "pcos"), _prepare_table(endo, "endo")],
        ignore_index=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    return combined
