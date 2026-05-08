import pandas as pd

from src.data.load_data import load_endometriosis, load_pcos
from src.utils.config import CONFIG


def test_load_pcos_non_empty():
	df = load_pcos()
	assert isinstance(df, pd.DataFrame)
	assert not df.empty
	assert CONFIG.pcos_target_column in df.columns


def test_load_endo_non_empty():
	df = load_endometriosis()
	assert isinstance(df, pd.DataFrame)
	assert not df.empty
	assert CONFIG.endo_target_column in df.columns
