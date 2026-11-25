"""Data cleaning: dtype modifications, column removals, missing value checks."""

import pandas as pd
from pathlib import Path

def clean_data(raw_path: Path, out_path: Path, config: dict) -> pd.DataFrame:
    """Apply cleaning rules defined in config."""
    clean_cfg = config["cleaning"]

    df = pd.read_excel(raw_path, index_col=None, skiprows=1)

    # ── Datetime columns (skip in numeric loop) ──────────────────
    DATETIME_COLUMNS = {"MonthYear"}

    # ── Numeric coercion with mean-fill ─────────────────────────
    for col in df.select_dtypes(include="object").columns:
        if col in DATETIME_COLUMNS:
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() > len(df) * 0.8:
            coerced_mask = df[col].notna() & converted.isna()
            col_mean = converted.mean()
            converted[coerced_mask] = col_mean
            df[col] = converted.astype("float64")

    # ── Datetime conversion ─────────────────────────────────────
    for col in DATETIME_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # ── Column Names conversion ─────────────────────────────────
    df = df.rename(columns={
        'Time index': 'TimeIndex', 'Dept_ID': 'DeptID',
        'Dept. Name': 'DeptName', 'Sales units': 'SalesUnits',
        'Area (m2)': 'Area(m2)', 'Opening hours': 'OpeningHours'
    })

    # ── 1. Drop explicitly listed columns (from config) ─────────
    drop_cols = clean_cfg.get("drop_columns", [])
    # # Also keep your hardcoded ID columns
    drop_cols = list(set(drop_cols + ['StoreID', 'DeptID', 'Customer']))
    drop_cols = [c for c in drop_cols if c in df.columns]  # safe filter
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"[cleaning] Dropped columns: {drop_cols}")

    # ── 2. Drop columns with too few unique values ──────────────
    min_unique = clean_cfg.get("min_unique_threshold", 2)
    low_unique = [
        col for col in df.columns
        if df[col].nunique() < min_unique
    ]
    if low_unique:
        df = df.drop(columns=low_unique)
        print(f"[cleaning] Dropped low-unique columns (<{min_unique}): {low_unique}")

    # ── 3. Drop columns with too many missing values ────────────
    max_missing = clean_cfg.get("max_missing_ratio", 0.5)
    missing_ratio = df.isnull().mean()
    high_missing = missing_ratio[missing_ratio > max_missing].index.tolist()
    if high_missing:
        df = df.drop(columns=high_missing)
        print(f"[cleaning] Dropped high-missing columns (>{max_missing:.0%}): {high_missing}")

    # ── Final summary ───────────────────────────────────────────
    print(f"[cleaning] Shape after cleaning: {df.shape}")
    saved_path = Path(out_path).joinpath("interim_cleaned.csv")
    df.to_csv(saved_path, index=False)
    print(f"[cleaning] Saved to {saved_path.as_posix()}")
    return df
