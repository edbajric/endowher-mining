"""Run preprocessing for PCOS and endometriosis datasets."""

from src.data.preprocess_endo import preprocess_endo
from src.data.preprocess_pcos import preprocess_pcos
from src.utils.config import CONFIG
from src.utils.io import save_dataframe


def _print_distribution(name, y):
    counts = y.value_counts(dropna=False)
    print(f"{name} target distribution:\n{counts}")


def main():
    xp_train, xp_test, yp_train, yp_test = preprocess_pcos()
    xe_train, xe_test, ye_train, ye_test = preprocess_endo()

    print("PCOS train/test:", xp_train.shape, xp_test.shape)
    _print_distribution("PCOS", yp_train)

    print("Endometriosis train/test:", xe_train.shape, xe_test.shape)
    _print_distribution("Endometriosis", ye_train)

    save_dataframe(xp_train, CONFIG.processed_data_dir / "pcos_x_train.csv")
    save_dataframe(xp_test, CONFIG.processed_data_dir / "pcos_x_test.csv")
    save_dataframe(
        yp_train.to_frame(CONFIG.pcos_target_column),
        CONFIG.processed_data_dir / "pcos_y_train.csv",
    )
    save_dataframe(
        yp_test.to_frame(CONFIG.pcos_target_column),
        CONFIG.processed_data_dir / "pcos_y_test.csv",
    )

    save_dataframe(xe_train, CONFIG.processed_data_dir / "endo_x_train.csv")
    save_dataframe(xe_test, CONFIG.processed_data_dir / "endo_x_test.csv")
    save_dataframe(
        ye_train.to_frame(CONFIG.endo_target_column),
        CONFIG.processed_data_dir / "endo_y_train.csv",
    )
    save_dataframe(
        ye_test.to_frame(CONFIG.endo_target_column),
        CONFIG.processed_data_dir / "endo_y_test.csv",
    )


if __name__ == "__main__":
    main()