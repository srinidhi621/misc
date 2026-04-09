from __future__ import annotations

from pathlib import Path

import pandas as pd

from .backtesting import rolling_origin_backtest
from .baseline import evaluate_naive_baseline
from .config import FORECAST_HORIZON_WEEKS, Paths
from .data_generation import persist_synthetic_data
from .decision import optimize_procurement
from .features import build_demand_features_asof, build_yield_features_asof
from .memo import generate_procurement_memo
from .models import train_quantile_bundle


def _latest_row_prediction(features: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    clean = features.dropna(subset=[target_col]).dropna(axis=0)
    feature_cols = [c for c in clean.columns if c not in {"week_start", target_col}]
    x_train = clean.iloc[:-1][feature_cols]
    y_train = clean.iloc[:-1][target_col]
    x_latest = clean.iloc[[-1]][feature_cols]
    meta = clean.iloc[[-1]][["week_start"]]
    return (x_train, y_train, x_latest, meta)


def run_pipeline(output_dir: Path | None = None) -> dict[str, pd.DataFrame | str]:
    paths = Paths()
    generated = persist_synthetic_data(paths=paths)

    sales = generated["sales"]
    yield_df = generated["yield"]
    costs = generated["costs"]

    demand_features = build_demand_features_asof(sales, horizon_weeks=FORECAST_HORIZON_WEEKS)
    yield_features = build_yield_features_asof(yield_df, horizon_weeks=FORECAST_HORIZON_WEEKS)

    demand_bt = rolling_origin_backtest(
        demand_features,
        target_col="target_demand_kg",
        min_train_size=90,
        step_size=26,
        n_folds=4,
    )
    yield_bt = rolling_origin_backtest(
        yield_features,
        target_col="target_yield_rate",
        min_train_size=90,
        step_size=26,
        n_folds=4,
    )

    x_d_train, y_d_train, x_d_latest, meta = _latest_row_prediction(demand_features, "target_demand_kg")
    x_y_train, y_y_train, x_y_latest, _ = _latest_row_prediction(yield_features, "target_yield_rate")

    demand_model = train_quantile_bundle(x_d_train, y_d_train)
    yield_model = train_quantile_bundle(x_y_train, y_y_train)

    demand_q = demand_model.predict(x_d_latest).iloc[0].to_dict()
    yield_q = yield_model.predict(x_y_latest).iloc[0].to_dict()

    cost_row = costs.iloc[-1]
    decision_grid, best = optimize_procurement(demand_q, yield_q, cost_row=cost_row)
    baseline = evaluate_naive_baseline(sales["demand_kg"], yield_df["yield_rate"], cost_row=cost_row)

    memo_payload = {
        "week_start": str(meta["week_start"].iloc[0].date()),
        "recommended_milk_liters": float(best["milk_liters"]),
        "expected_total_cost": float(best["expected_total_cost"]),
        "expected_milk_cost": float(best["expected_milk_cost"]),
        "expected_holding_cost": float(best["expected_holding_cost"]),
        "expected_stockout_cost": float(best["expected_stockout_cost"]),
        "expected_spoilage_cost": float(best["expected_spoilage_cost"]),
        "baseline_expected_total_cost": float(baseline["expected_total_cost"]),
        "estimated_savings_vs_baseline": float(baseline["expected_total_cost"] - best["expected_total_cost"]),
    }
    memo = generate_procurement_memo(memo_payload)

    out_dir = output_dir or (paths.project_root / "data" / "external")
    out_dir.mkdir(parents=True, exist_ok=True)
    demand_bt.fold_predictions.to_csv(out_dir / "demand_backtest_predictions.csv", index=False)
    yield_bt.fold_predictions.to_csv(out_dir / "yield_backtest_predictions.csv", index=False)
    decision_grid.to_csv(out_dir / "decision_cost_curve.csv", index=False)
    pd.DataFrame([memo_payload]).to_csv(out_dir / "procurement_recommendation.csv", index=False)
    (out_dir / "procurement_memo.md").write_text(memo)

    return {
        "demand_backtest": demand_bt.fold_predictions,
        "yield_backtest": yield_bt.fold_predictions,
        "decision_grid": decision_grid,
        "recommendation": pd.DataFrame([memo_payload]),
        "memo": memo,
    }


if __name__ == "__main__":
    run_pipeline()
