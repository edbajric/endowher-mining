from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    # Root and global settings
    root_dir: Path = Path(__file__).resolve().parents[2]
    random_state: int = 42
    test_size: float = 0.2

    # Targets (TODO: confirm exact column names)

    pcos_target_column: str = "PCOS (Y/N)"
    endo_target_column: str = "Diagnosis"

    # Filenames
    pcos_infertility_filename: str = "PCOS_infertility.csv"
    pcos_no_infertility_filename: str = "PCOS_data_without_infertility.xlsx"
    endo_default_filename: str = "structured_endometriosis_data.csv"

    # Raw data paths
    @property
    def pcos_raw_dir(self) -> Path:
        return self.root_dir / "data" / "raw" / "pcos"

    @property
    def endo_raw_dir(self) -> Path:
        return self.root_dir / "data" / "raw" / "endometriosis"

    @property
    def pcos_infertility_path(self) -> Path:
        return self.pcos_raw_dir / self.pcos_infertility_filename

    @property
    def pcos_no_infertility_path(self) -> Path:
        return self.pcos_raw_dir / self.pcos_no_infertility_filename

    @property
    def endo_raw_path(self) -> Path:
        return self.endo_raw_dir / self.endo_default_filename

    # Output directories
    @property
    def figures_dir(self) -> Path:
        return self.root_dir / "reports" / "figures"

    @property
    def tables_dir(self) -> Path:
        return self.root_dir / "reports" / "tables"

    @property
    def models_dir(self) -> Path:
        return self.root_dir / "models"

    @property
    def interim_data_dir(self) -> Path:
        return self.root_dir / "data" / "interim"

    @property
    def processed_data_dir(self) -> Path:
        return self.root_dir / "data" / "processed"


# Singleton config to import from other modules
CONFIG = ProjectConfig()