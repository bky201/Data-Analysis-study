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

if __name__ == "__main__":
    main()
