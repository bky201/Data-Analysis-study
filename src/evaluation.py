"""Evaluation: R-squared and RMSE calculation."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import numpy as np

from sklearn.base import clone

import pandas as pd
from src.models import build_models
from src.overfitting_diagonistic import compute_overfitting_metrics
from src.dim_reduction import reduce
from src.scaling import build_preprocessor


# ─── Final Evaluation (on held-out test set) ──────────────────────────────

def evaluate_final(X_train, X_test, y_train, y_test,
                   num_cols, cat_cols, cfg, results_df) -> dict:
    """Re-fit the top-2 pipelines on full train, evaluate on test."""
    all_models = build_models(cfg)
    final_results = {}

    for rank, label in [(0, "best"), (1, "runner_up")]:
        if rank >= len(results_df):
            break
        row = results_df.iloc[rank]
        scale_m, red_m, model_name = row["scaling"], row["reduction"], row["model"]

        cfg_local = {**cfg, "scaling": {**cfg["scaling"], "method": scale_m}}
        preprocessor = build_preprocessor(cfg_local, num_cols, cat_cols)
        X_tr_df = pd.DataFrame(X_train, columns=num_cols + cat_cols)
        X_te_df = pd.DataFrame(X_test, columns=num_cols + cat_cols)
        X_tr_s = np.asarray(preprocessor.fit_transform(X_tr_df))
        X_te_s = np.asarray(preprocessor.transform(X_te_df))

        cfg_local["dim_reduction"] = {**cfg["dim_reduction"], "method": red_m}
        X_tr_r, X_te_r, _ = reduce(X_tr_s, X_te_s, cfg_local)

        model = clone(all_models[model_name])
        model.fit(X_tr_r, y_train)
        y_tr_pred = model.predict(X_tr_r)
        y_te_pred = model.predict(X_te_r)

        metrics = compute_overfitting_metrics(y_train, y_tr_pred, y_test, y_te_pred)
        metrics["y_pred_test"] = y_te_pred
        metrics["y_pred_train"] = y_tr_pred
        final_results[label] = metrics

    return final_results
