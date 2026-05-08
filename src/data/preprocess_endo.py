from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_endo_preprocessor(x, scale_numeric: bool = False) -> ColumnTransformer:
    """Build preprocessing for endometriosis dataset.

    TODO: Confirm feature datatypes and any dataset-specific cleaning rules.
    """
    numeric_features = x.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [c for c in x.columns if c not in numeric_features]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
