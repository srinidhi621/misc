from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .models import pinball_loss, train_quantile_bundle


@dataclass
class BacktestResult:
    fold_predictions: pd.DataFrame
    aggregate_metrics: pd.DataFrame


def rolling_origin_backtest(
    feature_df: pd.DataFrame,
    target_col: str,
    min_train_size: int,
    step_size: int,
    n_folds: int,
) -> BacktestResult:
    df = feature_df.dropna(subset=[target_col]).dropna(axis=0).reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in {"week_start", target_col}]

    all_preds: list[pd.DataFrame] = []
    metrics: list[dict[str, float]] = []

    for fold in range(n_folds):
        train_end = min_train_size + fold * step_size
        test_start = train_end
        test_end = min(train_end + step_size, len(df))
        if test_end <= test_start:
            break

        train_df = df.iloc[:train_end]
        test_df = df.iloc[test_start:test_end]

        model = train_quantile_bundle(train_df[feature_cols], train_df[target_col])
        preds = model.predict(test_df[feature_cols])

        fold_pred = pd.DataFrame(
            {
                "week_start": test_df["week_start"].values,
                "y_true": test_df[target_col].values,
                "q10": preds["q10"].values,
                "q50": preds["q50"].values,
                "q90": preds["q90"].values,
                "fold": fold,
            }
        )
        all_preds.append(fold_pred)

        metrics.append(
            {
                "fold": fold,
                "pinball_q10": pinball_loss(fold_pred["y_true"].to_numpy(), fold_pred["q10"].to_numpy(), 0.1),
                "pinball_q50": pinball_loss(fold_pred["y_true"].to_numpy(), fold_pred["q50"].to_numpy(), 0.5),
                "pinball_q90": pinball_loss(fold_pred["y_true"].to_numpy(), fold_pred["q90"].to_numpy(), 0.9),
            }
        )

    pred_df = pd.concat(all_preds, ignore_index=True)
    metric_df = pd.DataFrame(metrics)
    agg = metric_df.mean(numeric_only=True).to_frame(name="value")

    return BacktestResult(fold_predictions=pred_df, aggregate_metrics=agg)
