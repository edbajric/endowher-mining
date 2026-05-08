from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    root_dir: Path = Path(__file__).resolve().parents[2]
    random_state: int = 42
    test_size: float = 0.2

    # TODO: Confirm exact column names in each dataset before training.
    pcos_target_column: str = "PCOS (Y/N)"
    endo_target_column: str = "TARGET_COLUMN_TO_CONFIRM"

    pcos_default_filename: str = "pcos.csv"
    endo_default_filename: str = "endometriosis.csv"

    @property
    def pcos_raw_path(self) -> Path:
        return self.root_dir / "data" / "raw" / "pcos" / self.pcos_default_filename

    @property
    def endo_raw_path(self) -> Path:
        return self.root_dir / "data" / "raw" / "endometriosis" / self.endo_default_filename

    @property
    def figures_dir(self) -> Path:
        return self.root_dir / "reports" / "figures"

    @property
    def tables_dir(self) -> Path:
        return self.root_dir / "reports" / "tables"

    @property
    def models_dir(self) -> Path:
        return self.root_dir / "models"
