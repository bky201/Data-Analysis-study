"""
main.py — End-to-end orchestration.

Pipeline: Clean → Impute → Scale → Select (Nested CV) → Evaluate → Statistics
"""
import sys
import yaml
import json
import logging
from pathlib import Path

import numpy as np

from src.sample_data import generate_sample_data
from src.cleaning import clean_data
from src.imputation import impute_data, save_imputed
from src.scaling import scale_all_methods
from src.pipeline_selection import run_nested_cv
from src.evaluation import evaluate_final
from src.statistical_test import (
    bootstrap_ci, bootstrap_difference,
    paired_t_test, friedman_test,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str = "config/config.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def main():
    cfg = load_config()

    """
    Adding a --smoke flag
    Speed in CI
    --smoke flag reduces data size and method count
    """

    if "--smoke" in sys.argv:
        cfg["imputation"]["n_imputations"] = 1
        cfg["dim_reduction"]["autoencoder"]["epochs"] = 10
        cfg["scaling"]["methods"] = ["standard"]  # only one method
        cfg["dim_reduction"]["methods"] = ["pca"]  # only one method
        # Use a tiny sample of data
        df = generate_sample_data(Path(cfg["data"]["sample_path"]), 100, 42)

    target_col = cfg["data"]["target_column"]
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("STAGE 1: Data Cleaning")
    logger.info("=" * 60)
    df = clean_data(
        Path(cfg["data"]["raw_path"]),
        Path(cfg["data"]["interim_path"]),
        config=cfg,
    )

    logger.info("=" * 60)
    logger.info("STAGE 2: Multiple Imputation")
    logger.info("=" * 60)
    all_imputed, pooled = impute_data(
        df,
        n_samples=cfg["imputation"]["n_imputations"],
        max_iter=cfg["imputation"]["max_iter"],
        impute_path=Path(cfg["data"]["interim_path"])
    )
    save_imputed(all_imputed, cfg["data"]["interim_path"])
    df = pooled

    logger.info("=" * 60)
    logger.info("STAGE 3: Scaling")
    logger.info("=" * 60)
    all_scaled, y_train, y_test = scale_all_methods(
        df, cfg, target_col, cfg["data"]["processed_path"]
    )
    X_train_base, X_test_base, _ = all_scaled[cfg["scaling"]["methods"][0]]

    X_df = df.drop(columns=[target_col], errors="ignore")
    num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_df.select_dtypes(include=["object", "str", "category"]).columns.tolist()

    logger.info("=" * 60)
    logger.info("STAGE 4: Automated Pipeline Selection (Nested CV)")
    logger.info("=" * 60)
    results_df = run_nested_cv(
        X_train_base, y_train, num_cols, cat_cols, cfg,
        n_splits=5, n_repeats=3,
    )

    results_df.drop(columns=["fold_scores"], errors="ignore").to_csv(
        results_dir / "comparison_summary.csv", index=False
    )

    logger.info("=" * 60)
    logger.info("STAGE 5: Final Evaluation (Held-Out Test)")
    logger.info("=" * 60)
    final = evaluate_final(
        X_train_base, X_test_base, y_train, y_test,
        num_cols, cat_cols, cfg, results_df,
    )

    best = final["best"]
    runner = final.get("runner_up")

    logger.info(f"\n  BEST:      R²={best['r2_test']:.4f}  RMSE={best['rmse_test']:.4f}  "
                f"[{best['diagnosis']}]")
    if runner:
        logger.info(f"  RUNNER-UP: R²={runner['r2_test']:.4f}  RMSE={runner['rmse_test']:.4f}  "
                    f"[{runner['diagnosis']}]")

    logger.info("=" * 60)
    logger.info("STAGE 6: Statistical Comparison")
    logger.info("=" * 60)

    ci_best = bootstrap_ci(y_test, best["y_pred_test"])
    logger.info(f"\n  Best R² = {ci_best['point']:.4f}  "
                f"95% CI [{ci_best['ci_95'][0]:.4f}, {ci_best['ci_95'][1]:.4f}]")

    if runner:
        diff = bootstrap_difference(y_test, best["y_pred_test"], runner["y_pred_test"])
        logger.info(f"  ΔR² (best − runner) = {diff['point_diff']:.4f}  "
                    f"95% CI [{diff['ci_95'][0]:.4f}, {diff['ci_95'][1]:.4f}]")
        logger.info(f"  Significant: {diff['significant']}")

    if "fold_scores" in results_df.columns:
        best_folds = results_df.iloc[0]["fold_scores"]
        runner_folds = results_df.iloc[1]["fold_scores"]
        if len(best_folds) == len(runner_folds):
            pt = paired_t_test(best_folds, runner_folds)
            logger.info(f"\n  Paired t-test: t={pt['t_stat']:.3f}, p={pt['p_value']:.4f}")
            logger.info(f"  Mean ΔR² = {pt['mean_diff']:.4f}  "
                        f"95% CI [{pt['ci_95'][0]:.4f}, {pt['ci_95'][1]:.4f}]")

    fold_scores_by_model = {}
    for model_name in results_df["model"].unique():
        subset = results_df[results_df["model"] == model_name]
        fold_scores_by_model[model_name] = subset["mean_r2"].tolist()

    if len(fold_scores_by_model) > 2:
        chi2, p_val = friedman_test(fold_scores_by_model)
        logger.info(f"\n  Friedman test: χ²={chi2:.3f}, p={p_val:.4f}")
        logger.info(f"  Models significantly different: {p_val < 0.05}")

    logger.info("=" * 60)
    logger.info("STAGE 7: Saving Results")
    logger.info("=" * 60)

    report = {
        "best_pipeline": {
            "scaling": results_df.iloc[0]["scaling"],
            "reduction": results_df.iloc[0]["reduction"],
            "model": results_df.iloc[0]["model"],
            "r2_test": best["r2_test"],
            "rmse_test": best["rmse_test"],
            "r2_ci": list(ci_best["ci_95"]),
            "diagnosis": best["diagnosis"],
        },
        "runner_up": None,
        "statistical_test": None,
    }

    if runner:
        report["runner_up"] = {
            "scaling": results_df.iloc[1]["scaling"],
            "reduction": results_df.iloc[1]["reduction"],
            "model": results_df.iloc[1]["model"],
            "r2_test": runner["r2_test"],
            "rmse_test": runner["rmse_test"],
            "diagnosis": runner["diagnosis"],
        }

    if "fold_scores" in results_df.columns:
        best_folds = results_df.iloc[0]["fold_scores"]
        runner_folds = results_df.iloc[1]["fold_scores"]
        if len(best_folds) == len(runner_folds):
            pt = paired_t_test(best_folds, runner_folds)
            report["statistical_test"] = {
                "t_stat": pt["t_stat"],
                "p_value": pt["p_value"],
                "significant": pt["significant"],
                "ci_95": list(pt["ci_95"]),
            }

    with open(results_dir / "final_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"\n[done] Final report → results/final_report.json")
    logger.info(f"[done] Comparison table → results/comparison_summary.csv")

if __name__ == "__main__":
    main()
