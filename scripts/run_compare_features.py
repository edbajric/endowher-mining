"""Compare feature importances between PCOS and endometriosis."""

from src.explain.compare_features import compare_feature_importance_tables
from src.utils.config import CONFIG
from src.utils.io import save_dataframe


def main() -> None:
	pcos_path = CONFIG.tables_dir / "pcos_feature_importances.csv"
	endo_path = CONFIG.tables_dir / "endo_feature_importances.csv"
	output_path = CONFIG.tables_dir / "shared_feature_comparison.csv"

	combined = compare_feature_importance_tables(
		pcos_table_path=pcos_path,
		endo_table_path=endo_path,
		output_path=output_path,
	)

	save_dataframe(combined, output_path)


if __name__ == "__main__":
	main()
