"""Train endometriosis models and save metrics, plots, and feature importances."""
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from src.data.preprocess_endo import preprocess_endo
from src.models.evaluate_models import evaluate_models
from src.models.train_logreg import train_logistic_regression
from src.models.train_random_forest import (
	extract_tree_feature_importances,
	train_decision_tree,
	train_gradient_boosting,
	train_random_forest,
	train_xgboost_optional,
)
from src.utils.config import CONFIG
from src.utils.io import save_dataframe
from src.utils.plotting import save_feature_importances


def _get_feature_names(pipeline, x_train: pd.DataFrame) -> list:
	preprocessor = pipeline.named_steps.get("preprocessor")
	if preprocessor is None:
		return list(x_train.columns)
	if hasattr(preprocessor, "get_feature_names_out"):
		return list(preprocessor.get_feature_names_out())
	return list(x_train.columns)


def _best_model_name(metrics: pd.DataFrame) -> str:
	if metrics["roc_auc"].notna().any():
		return metrics.sort_values("roc_auc", ascending=False).iloc[0]["model_name"]
	return metrics.sort_values("f1", ascending=False).iloc[0]["model_name"]


def _update_metrics_table(metrics: pd.DataFrame) -> pd.DataFrame:
	metrics = metrics.copy()
	best_name = _best_model_name(metrics)
	metrics["is_best"] = metrics["model_name"] == best_name

	table_path = CONFIG.tables_dir / "model_metrics.csv"
	if table_path.exists():
		existing = pd.read_csv(table_path)
		existing = existing[existing["dataset"] != "endo"]
		metrics = pd.concat([existing, metrics], ignore_index=True)

	save_dataframe(metrics, table_path)
	return metrics


def _save_feature_importances(
	model_name: str,
	pipeline,
	x_train: pd.DataFrame,
	dataset_label: str,
) -> Tuple[pd.DataFrame, Path]:
	feature_names = _get_feature_names(pipeline, x_train)
	importances = extract_tree_feature_importances(pipeline, feature_names)
	table = (
		pd.DataFrame(
			{"feature": list(importances.keys()), "importance": list(importances.values())}
		)
		.sort_values("importance", ascending=False)
		.reset_index(drop=True)
	)

	output_path = CONFIG.tables_dir / f"{dataset_label}_feature_importances.csv"
	save_dataframe(table, output_path)

	save_feature_importances(
		features=table["feature"],
		importances=table["importance"],
		title=f"{dataset_label.upper()} {model_name} Feature Importances",
		output_path=CONFIG.figures_dir / f"{dataset_label}_{model_name}_feature_importance.png",
	)
	return table, output_path


def _save_shap_summary(pipeline, x_train: pd.DataFrame, dataset_label: str) -> None:
	try:
		import shap
	except ImportError:
		return

	preprocessor = pipeline.named_steps.get("preprocessor")
	model = pipeline.named_steps.get("model")
	if preprocessor is None or model is None:
		return

	x_transformed = preprocessor.transform(x_train)
	feature_names = _get_feature_names(pipeline, x_train)
	try:
		explainer = shap.TreeExplainer(model)
		shap_values = explainer.shap_values(x_transformed)
		shap.summary_plot(
			shap_values,
			x_transformed,
			feature_names=feature_names,
			show=False,
		)
		output_path = CONFIG.figures_dir / f"{dataset_label}_shap_summary.png"
		output_path.parent.mkdir(parents=True, exist_ok=True)
		shap.plt.gcf().savefig(output_path, dpi=200, bbox_inches="tight")
		shap.plt.close()
	except Exception:
		return


def main() -> None:
	x_train, x_test, y_train, y_test = preprocess_endo()

	models: Dict[str, object] = {}
	models["logreg"] = train_logistic_regression(
		x_train,
		y_train,
		dataset_name="endometriosis",
		model_out=CONFIG.models_dir / "endo" / "logreg.joblib",
		random_state=CONFIG.random_state,
	)

	models["decision_tree"] = train_decision_tree(
		x_train,
		y_train,
		dataset_name="endometriosis",
		model_out=CONFIG.models_dir / "endo" / "decision_tree.joblib",
		random_state=CONFIG.random_state,
	)

	models["random_forest"] = train_random_forest(
		x_train,
		y_train,
		dataset_name="endometriosis",
		model_out=CONFIG.models_dir / "endo" / "random_forest.joblib",
		random_state=CONFIG.random_state,
	)

	try:
		models["xgboost"] = train_xgboost_optional(
			x_train,
			y_train,
			dataset_name="endometriosis",
			model_out=CONFIG.models_dir / "endo" / "xgboost.joblib",
			random_state=CONFIG.random_state,
		)
	except ImportError:
		models["gradient_boosting"] = train_gradient_boosting(
			x_train,
			y_train,
			dataset_name="endometriosis",
			model_out=CONFIG.models_dir / "endo" / "gradient_boosting.joblib",
			random_state=CONFIG.random_state,
		)

	metrics = evaluate_models(
		models=models,
		x_test=x_test,
		y_test=y_test,
		dataset_name="endo",
		figures_dir=CONFIG.figures_dir,
		tables_dir=CONFIG.tables_dir,
	)
	metrics = _update_metrics_table(metrics)

	tree_candidates = {k: v for k, v in models.items() if k != "logreg"}
	if tree_candidates:
		best_tree_name = _best_model_name(metrics[metrics["model_name"].isin(tree_candidates.keys())])
		best_tree = tree_candidates[best_tree_name]
		_save_feature_importances(best_tree_name, best_tree, x_train, "endo")
		_save_shap_summary(best_tree, x_train, "endo")


if __name__ == "__main__":
	main()
