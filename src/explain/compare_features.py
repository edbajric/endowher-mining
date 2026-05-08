from pathlib import Path

import pandas as pd

from src.data.feature_mapping import normalize_feature_name


def compare_feature_importance_tables(
    pcos_table_path: Path,
    endo_table_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Merge feature-importance summaries by normalized labels (post-training only)."""
    pcos = pd.read_csv(pcos_table_path)
    endo = pd.read_csv(endo_table_path)

    if "feature" not in pcos.columns:
        raise KeyError(
            f"'feature' column missing in PCOS table: {pcos_table_path}. "
            "Include a 'feature' column before running cross-dataset comparison."
        )
    if "feature" not in endo.columns:
        raise KeyError(
            f"'feature' column missing in endometriosis table: {endo_table_path}. "
            "Include a 'feature' column before running cross-dataset comparison."
        )

    pcos = pcos.copy()
    endo = endo.copy()
    pcos["feature_key"] = pcos["feature"].map(normalize_feature_name)
    endo["feature_key"] = endo["feature"].map(normalize_feature_name)

    merged = pcos.merge(
        endo,
        on="feature_key",
        how="inner",
        suffixes=("_pcos", "_endo"),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return merged
