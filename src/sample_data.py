# src/sample_data.py — generates a tiny synthetic dataset for CI
import pandas as pd
import numpy as np
from pathlib import Path

def generate_sample_data(path: Path, n_rows=100, seed=42):
    """Create a small synthetic CSV that mimics the real schema."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "TimeIndex": pd.date_range("2020-01-01", periods=n_rows, freq="D"),
        "MonthYear": pd.date_range("2020-01-01", periods=n_rows, freq="D").to_period("M"),
        "Country": rng.choice(["US", "UK", "DE", "FR"], n_rows),
        "City": rng.choice(["NYC", "London", "Berlin", "Paris"], n_rows),
        "DeptName": rng.choice(["Electronics", "Clothing", "Food"], n_rows),
        "OpeningHours": rng.choice(["Full", "Reduced", "Closed"], n_rows),
        "SalesUnits": rng.poisson(50, n_rows).astype(float),
        "Area_m2": rng.uniform(50, 500, n_rows),
        "target": rng.normal(100, 20, n_rows),
    })
    # Inject some missing values
    df.loc[rng.choice(n_rows, 10, replace=False), "SalesUnits"] = np.nan
    df.loc[rng.choice(n_rows, 5, replace=False), "Country"] = np.nan

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[sample] Generated {n_rows} rows → {path}")
