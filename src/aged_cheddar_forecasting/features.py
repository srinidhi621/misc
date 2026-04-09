from __future__ import annotations

import numpy as np
import pandas as pd


def _asof_lagged_feature(
    df: pd.DataFrame,
    value_cols: list[str],
    lags: list[int],
    rolling_windows: list[int],
) -> pd.DataFrame:
    out = df.copy().sort_values("week_start").reset_index(drop=True)

    for col in value_cols:
        for lag in lags:
            out[f"{col}_lag_{lag}"] = out[col].shift(lag)
        for window in rolling_windows:
            out[f"{col}_rollmean_{window}"] = out[col].shift(1).rolling(window=window, min_periods=window).mean()

    return out


def build_demand_features_asof(sales_df: pd.DataFrame, horizon_weeks: int = 26) -> pd.DataFrame:
    """Leakage-safe demand features.

    All lagged/rolling targets are shifted, so features for week t only use information <= t-1.
    Target is demand at t + horizon_weeks.
    """
    df = sales_df.sort_values("week_start").reset_index(drop=True)
    df = _asof_lagged_feature(df, ["demand_kg"], lags=[1, 2, 4, 8, 13], rolling_windows=[4, 8, 13])
    df["week_of_year"] = df["week_start"].dt.isocalendar().week.astype(int)
    df["sin_woy"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["cos_woy"] = np.cos(2 * np.pi * df["week_of_year"] / 52)

    df["target_demand_kg"] = df["demand_kg"].shift(-horizon_weeks)
    return df


def build_yield_features_asof(yield_df: pd.DataFrame, horizon_weeks: int = 26) -> pd.DataFrame:
    """Leakage-safe yield features for forecasting future conversion efficiency."""
    df = yield_df.sort_values("week_start").reset_index(drop=True)
    df = _asof_lagged_feature(df, ["yield_rate", "fat_pct", "protein_pct"], lags=[1, 2, 4, 8], rolling_windows=[4, 8, 13])
    df["week_of_year"] = df["week_start"].dt.isocalendar().week.astype(int)
    df["sin_woy"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["cos_woy"] = np.cos(2 * np.pi * df["week_of_year"] / 52)

    df["target_yield_rate"] = df["yield_rate"].shift(-horizon_weeks)
    return df


def model_matrix(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    clean = df.dropna(subset=[target_col]).dropna(axis=0)
    feature_cols = [c for c in clean.columns if c not in {"week_start", target_col}]
    return clean[feature_cols], clean[target_col]
