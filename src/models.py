"""Model definitions: XGBoost, MLP, Random Forest."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from sklearn.base import BaseEstimator

def build_models(config: dict) -> dict[str, BaseEstimator]:
    """Instantiate all models from config."""
    models_cfg = config["models"]
    models = {}

    models["xgboost"] = xgb.XGBRegressor(
        n_estimators=models_cfg["xgboost"]["n_estimators"],
        max_depth=models_cfg["xgboost"]["max_depth"],
        learning_rate=models_cfg["xgboost"]["learning_rate"],
        random_state=models_cfg["xgboost"]["random_state"],
        eval_metric="rmse",
    )

    models["random_forest"] = RandomForestRegressor(
        n_estimators=models_cfg["random_forest"]["n_estimators"],
        max_depth=models_cfg["random_forest"]["max_depth"],
        random_state=models_cfg["random_forest"]["random_state"],
        n_jobs=-1,
    )

    mlp_cfg = models_cfg["mlp"]
    models["mlp"] = MLPRegressor(
        hidden_layer_sizes=tuple(mlp_cfg["hidden_layer_sizes"]),
        activation=mlp_cfg["activation"],
        max_iter=mlp_cfg["max_iter"],
        random_state=mlp_cfg["random_state"],
        early_stopping=True,
        validation_fraction=0.1,
    )

    print(f"[models] Built {len(models)} models: {list(models.keys())}")
    return models

def train_and_predict(models: dict, X_train: np.ndarray,
                      X_test: np.ndarray, y_train: np.ndarray,
                      y_test: np.ndarray) -> dict[str, np.ndarray]:
    """Train all models and return predictions."""
    predictions = {}
    for name, model in models.items():
        print(f"[models] Training {name}...")
        model.fit(X_train, y_train)
        predictions[name] = model.predict(X_test)
        print(f"[models] {name} trained.")
    return predictions
