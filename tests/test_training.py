from src.data.preprocess_pcos import preprocess_pcos
from src.models.train_logreg import train_logistic_regression
from src.models.train_random_forest import train_random_forest
from src.utils.metrics import compute_classification_metrics


def test_training_logreg_metrics(tmp_path):
	x_train, x_test, y_train, y_test = preprocess_pcos()
	x_train = x_train.head(200)
	y_train = y_train.head(200)
	x_test = x_test.head(100)
	y_test = y_test.head(100)

	model = train_logistic_regression(
		x_train,
		y_train,
		dataset_name="pcos",
		model_out=tmp_path / "logreg.joblib",
		random_state=42,
	)

	y_pred = model.predict(x_test)
	y_prob = model.predict_proba(x_test)[:, 1]
	metrics = compute_classification_metrics(y_test, y_pred, y_prob=y_prob)

	assert "accuracy" in metrics
	assert "precision" in metrics
	assert "recall" in metrics
	assert "f1" in metrics
	assert "roc_auc" in metrics


def test_training_random_forest_metrics(tmp_path):
	x_train, x_test, y_train, y_test = preprocess_pcos()
	x_train = x_train.head(200)
	y_train = y_train.head(200)
	x_test = x_test.head(100)
	y_test = y_test.head(100)

	model = train_random_forest(
		x_train,
		y_train,
		dataset_name="pcos",
		model_out=tmp_path / "rf.joblib",
		random_state=42,
		n_estimators=50,
	)

	y_pred = model.predict(x_test)
	y_prob = model.predict_proba(x_test)[:, 1]
	metrics = compute_classification_metrics(y_test, y_pred, y_prob=y_prob)

	assert "accuracy" in metrics
	assert "precision" in metrics
	assert "recall" in metrics
	assert "f1" in metrics
	assert "roc_auc" in metrics
