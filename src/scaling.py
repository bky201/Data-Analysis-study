"""Scaling: Min-Max, Z-score, RobustScaler + OneHot for categoricals."""

from sklearn.model_selection import train_test_split
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    MinMaxScaler, StandardScaler, RobustScaler, OrdinalEncoder
)
from sklearn.model_selection import train_test_split


def build_preprocessor(config: dict, num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    """Build a ColumnTransformer that scales numerics and one-hot encodes categoricals."""
    method = config["scaling"].get("method", "standard")
    scalers = {"minmax": MinMaxScaler, "standard": StandardScaler, "robust": RobustScaler}

    if method not in scalers:
        raise ValueError(f"Unknown scaling method: {method}")

    transformers = []

    if num_cols:
        transformers.append(("num", scalers[method](), num_cols))

    if cat_cols:
        transformers.append(
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols)
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def scale_all_methods(df: pd.DataFrame, config: dict, target_col: str, output_dir: str):

    """Apply ALL scaling methods, save each, return dict of results."""

    eval_cfg = config["evaluation"]
    test_size = eval_cfg.get("test_size", 0.2)
    rs = eval_cfg.get("random_state", 42)

    if target_col in df.columns:
        y = df[target_col]
        X = df.drop(columns=[target_col], errors="ignore")
        print(y.iloc[0])
    else:
        print("Target column missing. Cannot proceed.")

    idx_train, idx_test = train_test_split(
    range(len(X)),
    test_size=test_size,
    random_state=rs
)
    X_train = X.iloc[idx_train].reset_index(drop=True)
    X_test = X.iloc[idx_test].reset_index(drop=True)
    y_train = y.iloc[idx_train].to_numpy()
    y_test = y.iloc[idx_test].to_numpy()

    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if y is not None:
        y =  y.to_numpy()
        np.save(Path(output_dir) / "y.npy", y)

    results = {}
    for method in config["scaling"]["methods"]:
        config["scaling"]["method"] = method
        preprocessor = build_preprocessor(config, num_cols, cat_cols)
        X_train_s = np.asarray(preprocessor.fit_transform(X_train))
        X_test_s = np.asarray(preprocessor.transform(X_test))

        feature_names = preprocessor.get_feature_names_out()
        print(feature_names)

        np.save(Path(output_dir) / f"X_train_{method}.npy", X_train_s)
        np.save(Path(output_dir) / f"X_test_{method}.npy", X_test_s)

        results[method] = (X_train_s, X_test_s, preprocessor)
        print(f"[scaling] {method} → {X_train_s.shape}")

    return results, y_train, y_test

def save_processed(X_train: np.ndarray, X_test: np.ndarray,
                   y, output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    np.save(Path(output_dir) / "X_train.npy", X_train)
    np.save(Path(output_dir) / "X_test.npy", X_test)
    if y is not None:
        np.save(Path(output_dir) / "y.npy", y.values if hasattr(y, "values") else np.array(y))
    print(f"[scaling] Saved processed matrices to {output_dir}/")
