"""Run the full workflow: preprocessing, training, comparison, reporting."""

from scripts.run_compare_features import main as compare_main
from scripts.run_make_report import main as report_main
from scripts.run_preprocessing import main as preprocess_main
from scripts.run_train_endo import main as train_endo_main
from scripts.run_train_pcos import main as train_pcos_main


def main() -> None:
	preprocess_main()
	train_pcos_main()
	train_endo_main()
	compare_main()
	report_main()


if __name__ == "__main__":
	main()
