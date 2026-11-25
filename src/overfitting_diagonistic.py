# ─── Overfitting Diagnostics ──────────────────────────────────────────────
import numpy as np

from sklearn.metrics import r2_score, mean_squared_error

def diagnose_overfitting(r2_train: float, r2_test: float,
                         rmse_train: float, rmse_test: float) -> str:
    """Classify fit quality based on train-test gap."""
    r2_gap = r2_train - r2_test
    rmse_ratio = rmse_test / max(rmse_train, 1e-8)

    if r2_train < 0.3 and r2_test < 0.3:
        return "UNDERFIT"
    elif r2_gap > 0.3 or rmse_ratio > 2.0:
        return "SEVERE_OVERFIT"
    elif r2_gap > 0.15 or rmse_ratio > 1.3:
        return "MODERATE_OVERFIT"
    elif r2_gap > 0.05:
        return "MILD_OVERFIT"
    else:
        return "GOOD_FIT"


def compute_overfitting_metrics(y_true, y_pred, X_train=None, X_test=None,
                                model=None) -> dict:
    """Compute train/test gap metrics."""
    r2_test = r2_score(y_true, y_pred)
    rmse_test = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    if X_train is not None and model is not None:
        y_train_pred = model.predict(X_train)
        r2_train = r2_score(X_train[:, 0] * 0 + y_true[:len(y_train_pred)] if False else
                            _get_y_train(y_true, model), y_train_pred)
        r2_train = r2_score(model._y_train if hasattr(model, '_y_train') else y_true,
                            y_train_pred) if hasattr(model, '_y_train') else r2_test
        rmse_train = float(np.sqrt(mean_squared_error(
            model._y_train if hasattr(model, '_y_train') else y_true,
            y_train_pred
        )))
    else:
        r2_train = r2_test
        rmse_train = rmse_test

    return {
        "r2_train": r2_train,
        "r2_test": r2_test,
        "rmse_train": rmse_train,
        "rmse_test": rmse_test,
        "r2_gap": r2_train - r2_test,
        "rmse_ratio": rmse_test / max(rmse_train, 1e-8),
        "diagnosis": diagnose_overfitting(r2_train, r2_test, rmse_train, rmse_test),
    }


def _get_y_train(y_test, model):
    """Placeholder — in practice, pass y_train explicitly."""
    return y_test
