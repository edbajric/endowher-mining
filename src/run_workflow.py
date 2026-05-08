from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.load_data import load_csv, split_features_target
from src.models.evaluate_models import evaluate_binary_model
from src.models.train_logreg import train_logistic_regression
from src.models.train_random_forest import train_random_forest
from src.utils.config import ProjectConfig


def _run_dataset_pipeline(df: pd.DataFrame, target_column: str, dataset_name: str, cfg: ProjectConfig):
    x, y = split_features_target(df=df, target_column=target_column)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    logreg_model = train_logistic_regression(
        x_train=x_train,
        y_train=y_train,
        dataset_name=dataset_name,
        model_out=cfg.models_dir / f"{dataset_name}_logreg.joblib",
        random_state=cfg.random_state,
    )
    rf_model = train_random_forest(
        x_train=x_train,
        y_train=y_train,
        dataset_name=dataset_name,
        model_out=cfg.models_dir / f"{dataset_name}_rf.joblib",
        random_state=cfg.random_state,
    )

    evaluate_binary_model(
        model=logreg_model,
        x_test=x_test,
        y_test=y_test,
        model_name="logreg",
        dataset_name=dataset_name,
        figures_dir=cfg.figures_dir,
        tables_dir=cfg.tables_dir,
    )
    evaluate_binary_model(
        model=rf_model,
        x_test=x_test,
        y_test=y_test,
        model_name="random_forest",
        dataset_name=dataset_name,
        figures_dir=cfg.figures_dir,
        tables_dir=cfg.tables_dir,
    )


def main():
    cfg = ProjectConfig()

    # TODO: Confirm exact file names and target columns before running on real data.
    pcos_df = load_csv(Path(cfg.pcos_raw_path))
    endo_df = load_csv(Path(cfg.endo_raw_path))

    _run_dataset_pipeline(pcos_df, cfg.pcos_target_column, "pcos", cfg)
    _run_dataset_pipeline(endo_df, cfg.endo_target_column, "endometriosis", cfg)

    print("Pipelines complete. Compare outputs in reports/tables and reports/figures.")


if __name__ == "__main__":
    main()
