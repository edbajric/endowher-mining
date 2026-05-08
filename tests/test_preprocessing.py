from src.data.preprocess_endo import preprocess_endo
from src.data.preprocess_pcos import preprocess_pcos


def test_preprocess_pcos_splits_and_no_nan():
	x_train, x_test, y_train, y_test = preprocess_pcos()
	assert not x_train.empty
	assert not x_test.empty
	assert x_train.isna().sum().sum() == 0
	assert x_test.isna().sum().sum() == 0
	assert y_train.isna().sum() == 0
	assert y_test.isna().sum() == 0


def test_preprocess_endo_splits_and_no_nan():
	x_train, x_test, y_train, y_test = preprocess_endo()
	assert not x_train.empty
	assert not x_test.empty
	assert x_train.isna().sum().sum() == 0
	assert x_test.isna().sum().sum() == 0
	assert y_train.isna().sum() == 0
	assert y_test.isna().sum() == 0
