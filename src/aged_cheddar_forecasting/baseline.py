from __future__ import annotations

import pandas as pd

from .decision import optimize_procurement


def naive_quantiles_from_history(series: pd.Series, lookback: int = 26) -> dict[str, float]:
    hist = series.tail(lookback)
    return {
        "q10": float(hist.quantile(0.1)),
        "q50": float(hist.quantile(0.5)),
        "q90": float(hist.quantile(0.9)),
    }


def evaluate_naive_baseline(
    demand_history: pd.Series,
    yield_history: pd.Series,
    cost_row: pd.Series,
) -> pd.Series:
    demand_q = naive_quantiles_from_history(demand_history)
    yield_q = naive_quantiles_from_history(yield_history)
    _, optimal = optimize_procurement(demand_q, yield_q, cost_row=cost_row, n_samples=2500)
    return optimal
