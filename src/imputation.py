"""Multiple Imputation implementations (MICE, KNN)."""

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path("config/config.yaml").read_text())
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def impute_data(df: pd.DataFrame, n_samples: int, max_iter: int, impute_path: Path):
    """Apply cleaning rules defined in config."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    enc_dict = {}
    df_cat = df[cat_cols].copy()
    for col in cat_cols:
        categories=df[~df[col].isna()][col].unique().tolist()

        enc = OrdinalEncoder(categories=[categories], handle_unknown='use_encoded_value', unknown_value=-1)
        nonulls = df.loc[~df[col].isna()][col].to_numpy().reshape(-1, 1)
        enc.fit(nonulls)
        enc_dict[col] = enc
        df_cat.loc[~df_cat[col].isna(), [col]] = enc.transform(nonulls.reshape(-1, 1)).ravel().astype(str)

    print(f"[encoded clean] Shape after cleaning: {df_cat.shape}")
    Path(impute_path).mkdir(parents=True, exist_ok=True)
    df_cat.to_csv(Path(impute_path).joinpath("interim_encoded.csv"), index=False)
    print(f"[encoded pre-imputation] Saved to {impute_path}")

    col_indices = {col: i for i, col in enumerate(cat_cols)}

    original_modes = {}
    for col in cat_cols:
        vals = df[col].dropna()
        original_modes[col] = vals.mode().iloc[0]

    all_imputed = []

    for i in range(n_samples):
        work = df.copy()
        cat_mask = work[cat_cols].isna().values  # capture BEFORE imputation

        # ── Encode categoricals ──
        for col in cat_cols:
            work[col] = enc_dict[col].transform(
                work[col].to_numpy().reshape(-1, 1)
            ).ravel()

        # ── Categorical MICE ──
        if cat_cols:
            print("sample-", i)
            print("start", cat_cols)
            imp_cat = IterativeImputer(
                estimator=RandomForestClassifier(n_estimators=100, random_state=42 + i),
                max_iter=max_iter,
                random_state=100 + i,
                initial_strategy='most_frequent',
            )
            work[cat_cols] = imp_cat.fit_transform(work[cat_cols])
            print("completed", cat_cols)

            work[cat_cols] += np.random.normal(0, 0.1, work[cat_cols].shape) * cat_mask

            # Sanitize → valid int indices
            work_arr = work[cat_cols].values
            print("start-sanitize", cat_cols)
            for col in cat_cols:
                ci = col_indices[col]
                n_cats = len(enc_dict[col].categories_[0])
                vals = work_arr[:, ci]
                vals[vals < 0] = 0
                work_arr[:, ci] = np.clip(np.round(vals).astype(int), 0, n_cats - 1)
            work[cat_cols] = work_arr
            print("complete-sanitize", cat_cols)

            # Inverse-transform back to categories
            print("start-inverse", cat_cols)
            for col in cat_cols:
                ci = col_indices[col]
                work[col] = enc_dict[col].inverse_transform(
                    work[cat_cols].values[:, ci].reshape(-1, 1)
                ).ravel()
            print("complete-inverse", cat_cols)

        # ── Numerical MICE ──
        if num_cols:
            print("sample-", i)
            print("start", num_cols)
            num_mask = work[num_cols].isna().values  # capture BEFORE
            imp_num = IterativeImputer(
                estimator=RandomForestRegressor(n_estimators=100, random_state=42 + i),
                max_iter=max_iter,
                random_state=200 + i,
                initial_strategy='mean',
            )
            work[num_cols] = imp_num.fit_transform(work[num_cols])
            print("complete", num_cols)

            work[num_cols] += np.random.normal(0, 0.1, work[num_cols].shape) * num_mask

        all_imputed.append(work)

    pooled = pd.DataFrame(index=df.index)
    for col in cat_cols:
        stacked = np.array([d[col].values for d in all_imputed])
        pooled[col] = [Counter(stacked[:, j]).most_common(1)[0][0]
                       for j in range(stacked.shape[1])]
    for col in num_cols:
        stacked = np.array([d[col].values for d in all_imputed], dtype=float)
        pooled[col] = stacked.mean(axis=0)

    # ── Final summary ───────────────────────────────────────────
    print(f"[imputation] Shape after cleaning: {pooled.shape}")
    Path(impute_path).mkdir(parents=True, exist_ok=True)
    pooled.to_csv(Path(impute_path).joinpath("interim_imputed.csv"), index=False)
    print(f"[imputation] Saved to {impute_path}")
    return all_imputed, pooled

def save_imputed(dfs: list[pd.DataFrame], interim_path: str):
    Path(interim_path).mkdir(parents=True, exist_ok=True)
    for i, df in enumerate(dfs):
        df.to_csv(Path(interim_path).joinpath(f"imputed_{i}.csv"), index=False)
    print(f"[imputation] Saved {len(dfs)} files to {interim_path}")
