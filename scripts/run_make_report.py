"""Generate summary Markdown reports from metrics tables."""

import pandas as pd

from src.utils.config import CONFIG
from src.utils.io import ensure_dir


def _summarize_dataset(metrics: pd.DataFrame, dataset: str) -> str:
	subset = metrics[metrics["dataset"] == dataset]
	if subset.empty:
		return f"# {dataset.upper()} Results\n\nNo metrics available.\n"

	best = subset.sort_values("roc_auc", ascending=False).iloc[0]
	lines = [
		f"# {dataset.upper()} Results",
		"",
		f"Best model: {best['model_name']}",
		"",
		"Metrics:",
		f"- Accuracy: {best['accuracy']:.4f}",
		f"- Precision: {best['precision']:.4f}",
		f"- Recall: {best['recall']:.4f}",
		f"- F1: {best['f1']:.4f}",
		f"- ROC-AUC: {best['roc_auc']:.4f}",
		"",
	]
	return "\n".join(lines)


def main() -> None:
	metrics_path = CONFIG.tables_dir / "model_metrics.csv"
	if not metrics_path.exists():
		print("No metrics table found. Run training first.")
		return

	metrics = pd.read_csv(metrics_path)
	ensure_dir(CONFIG.root_dir / "reports" / "summary")

	pcos_summary = _summarize_dataset(metrics, "pcos")
	endo_summary = _summarize_dataset(metrics, "endo")

	(CONFIG.root_dir / "reports" / "summary" / "pcos_results.md").write_text(pcos_summary)
	(CONFIG.root_dir / "reports" / "summary" / "endo_results.md").write_text(endo_summary)


if __name__ == "__main__":
	main()
