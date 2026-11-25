"""Automated pipeline selection via repeated k-fold CV."""

import logging
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from sklearn.base import clone

from src.scaling import build_preprocessor
from src.dim_reduction import reduce, _fit_autoencoder
from src.models import build_models
from src.overfitting_diagonistic import diagnose_overfitting
from sklearn.metrics import r2_score

logger = logging.getLogger(__name__)
_ae_cache = {}

def _get_cached_ae(X_train, config):
    """Train AE once per (input_dim, config) combo; reuse across folds."""
    key = (X_train.shape[1], config["dim_reduction"]["n_components"],
           tuple(config["dim_reduction"]["autoencoder"]["hidden_dims"]))

    if key not in _ae_cache:
        X_enc, encode_fn, epochs = _fit_autoencoder(X_train, config)
        _ae_cache[key] = encode_fn
        print(f"    [AE cache] Trained new AE (key={key}), {epochs} epochs")
    else:
        encode_fn = _ae_cache[key]
        print(f"    [AE cache] Reusing cached AE (key={key})")

    return encode_fn

def run_nested_cv(X_train: np.ndarray, y_train: np.ndarray,
                  num_cols: list, cat_cols: list, cfg: dict,
                  n_splits: int = 5, n_repeats: int = 3) -> pd.DataFrame:
    """
    Automated pipeline selection via repeated k-fold CV.
    Tries all (scaling × reduction × model) combinations.
    Returns a DataFrame sorted by mean R².
    """
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    all_models = build_models(cfg)
    results = []

    scale_methods = cfg["scaling"]["methods"]
    red_methods = cfg["dim_reduction"]["methods"]
    model_names = list(all_models.keys())

    total = len(scale_methods) * len(red_methods) * len(model_names)
    run_idx = 0

    for scale_m, red_m, model_name in itertools.product(scale_methods, red_methods, model_names):
        run_idx += 1
        fold_scores = []
        train_scores = []

        for train_idx, val_idx in cv.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            cfg_local = {**cfg, "scaling": {**cfg["scaling"], "method": scale_m}}
            preprocessor = build_preprocessor(cfg_local, num_cols, cat_cols)
            X_tr_df = pd.DataFrame(X_tr, columns=num_cols + cat_cols)
            X_val_df = pd.DataFrame(X_val, columns=num_cols + cat_cols)
            X_tr_s = np.asarray(preprocessor.fit_transform(X_tr_df))
            X_val_s = np.asarray(preprocessor.transform(X_val_df))

            cfg_local["dim_reduction"] = {**cfg["dim_reduction"], "method": red_m}
            try:
                X_tr_r, X_val_r, _ = reduce(X_tr_s, X_val_s, cfg_local)
            except Exception as e:
                logger.warning(f"  [skip] {scale_m}/{red_m}/{model_name}: {e}")
                break

            model = clone(all_models[model_name])
            model.fit(X_tr_r, y_tr)
            y_val_pred = model.predict(X_val_r)
            y_tr_pred = model.predict(X_tr_r)

            fold_scores.append(r2_score(y_val, y_val_pred))
            train_scores.append(r2_score(y_tr, y_tr_pred))

        if not fold_scores:
            continue

        mean_r2 = np.mean(fold_scores)
        std_r2 = np.std(fold_scores)
        mean_r2_train = np.mean(train_scores)
        r2_gap = mean_r2_train - mean_r2

        results.append({
            "scaling": scale_m,
            "reduction": red_m,
            "model": model_name,
            "mean_r2": mean_r2,
            "std_r2": std_r2,
            "mean_r2_train": mean_r2_train,
            "r2_gap": r2_gap,
            "diagnosis": diagnose_overfitting(mean_r2_train, mean_r2, 1, 1 + r2_gap),
            "fold_scores": fold_scores,
        })

        logger.info(f"[{run_idx:2d}/{total}] {scale_m:8s} | {red_m:12s} | "
                    f"{model_name:14s} → R²={mean_r2:.4f}±{std_r2:.4f}  "
                    f"gap={r2_gap:.4f} ({results[-1]['diagnosis']})")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("mean_r2", ascending=False).reset_index(drop=True)
    return results_df
