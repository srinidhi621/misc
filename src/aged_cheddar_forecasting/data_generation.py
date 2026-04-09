from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from .config import CostAssumptions, Paths, RANDOM_SEED


def _week_index(n_weeks: int = 208) -> pd.DatetimeIndex:
    return pd.date_range("2022-01-03", periods=n_weeks, freq="W-MON")


def generate_weekly_sales_data(n_weeks: int = 208, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    weeks = _week_index(n_weeks)
    t = np.arange(n_weeks)

    promo = rng.binomial(1, 0.18, n_weeks)
    competitor_price_index = 1.0 + 0.03 * np.sin(2 * np.pi * t / 13) + rng.normal(0, 0.01, n_weeks)
    holiday_flag = ((weeks.month == 11) | (weeks.month == 12)).astype(int)
    avg_temp_c = 12 + 10 * np.sin(2 * np.pi * t / 52) + rng.normal(0, 1.5, n_weeks)

    trend = 0.55 * t
    seasonality = 40 * np.sin(2 * np.pi * t / 52)
    demand = (
        820
        + trend
        + seasonality
        + 70 * promo
        + 55 * holiday_flag
        + 45 * (competitor_price_index - 1.0)
        - 2.2 * avg_temp_c
        + rng.normal(0, 28, n_weeks)
    )

    return pd.DataFrame(
        {
            "week_start": weeks,
            "promo_intensity": promo,
            "competitor_price_index": competitor_price_index.round(4),
            "holiday_flag": holiday_flag,
            "avg_temp_c": avg_temp_c.round(2),
            "demand_kg": np.clip(demand, 500, None).round(2),
        }
    )


def generate_weekly_yield_data(n_weeks: int = 208, seed: int = RANDOM_SEED + 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    weeks = _week_index(n_weeks)
    t = np.arange(n_weeks)

    fat_pct = 3.7 + 0.12 * np.sin(2 * np.pi * t / 26) + rng.normal(0, 0.06, n_weeks)
    protein_pct = 3.3 + 0.08 * np.sin(2 * np.pi * t / 39 + 0.2) + rng.normal(0, 0.05, n_weeks)
    somatic_cell_count = np.clip(180 + 20 * np.sin(2 * np.pi * t / 52 + 1.8) + rng.normal(0, 15, n_weeks), 120, 300)
    pasture_quality_index = np.clip(70 + 18 * np.sin(2 * np.pi * t / 52 - 0.8) + rng.normal(0, 5, n_weeks), 40, 100)

    yield_rate = (
        0.105
        + 0.012 * (fat_pct - 3.7)
        + 0.007 * (protein_pct - 3.3)
        - 0.00003 * (somatic_cell_count - 180)
        + 0.00018 * (pasture_quality_index - 70)
        + rng.normal(0, 0.0015, n_weeks)
    )

    milk_collected_liters = 11500 + 320 * np.sin(2 * np.pi * t / 52 + 0.4) + rng.normal(0, 220, n_weeks)
    cheese_output_kg = milk_collected_liters * yield_rate

    return pd.DataFrame(
        {
            "week_start": weeks,
            "fat_pct": fat_pct.round(4),
            "protein_pct": protein_pct.round(4),
            "somatic_cell_count": somatic_cell_count.round(2),
            "pasture_quality_index": pasture_quality_index.round(2),
            "milk_collected_liters": milk_collected_liters.round(2),
            "yield_rate": yield_rate.round(5),
            "cheese_output_kg": cheese_output_kg.round(2),
        }
    )


def generate_weekly_cost_data(n_weeks: int = 208, seed: int = RANDOM_SEED + 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    weeks = _week_index(n_weeks)
    t = np.arange(n_weeks)
    base = CostAssumptions()

    milk_cost = base.milk_cost_per_liter + 0.03 * np.sin(2 * np.pi * t / 26) + rng.normal(0, 0.01, n_weeks)
    holding_cost = base.holding_cost_per_kg + 0.015 * np.cos(2 * np.pi * t / 52) + rng.normal(0, 0.005, n_weeks)
    stockout_cost = base.stockout_cost_per_kg + 0.08 * np.sin(2 * np.pi * t / 52 + 0.5) + rng.normal(0, 0.03, n_weeks)
    spoilage_cost = base.spoilage_cost_per_kg + 0.02 * np.cos(2 * np.pi * t / 39) + rng.normal(0, 0.01, n_weeks)

    df = pd.DataFrame(
        {
            "week_start": weeks,
            "milk_cost_per_liter": np.clip(milk_cost, 0.45, None).round(4),
            "holding_cost_per_kg": np.clip(holding_cost, 0.05, None).round(4),
            "stockout_cost_per_kg": np.clip(stockout_cost, 1.5, None).round(4),
            "spoilage_cost_per_kg": np.clip(spoilage_cost, 0.15, None).round(4),
            "spoilage_rate": base.spoilage_rate,
        }
    )
    return df


def persist_synthetic_data(paths: Paths = Paths(), n_weeks: int = 208) -> dict[str, pd.DataFrame]:
    paths.raw_data_dir.mkdir(parents=True, exist_ok=True)
    paths.external_data_dir.mkdir(parents=True, exist_ok=True)

    sales = generate_weekly_sales_data(n_weeks=n_weeks)
    yield_df = generate_weekly_yield_data(n_weeks=n_weeks)
    costs = generate_weekly_cost_data(n_weeks=n_weeks)

    sales.to_csv(paths.raw_data_dir / "weekly_sales_demand_drivers.csv", index=False)
    yield_df.to_csv(paths.raw_data_dir / "weekly_milk_yield_drivers.csv", index=False)
    costs.to_csv(paths.external_data_dir / "weekly_cost_assumptions.csv", index=False)

    assumptions = pd.DataFrame([asdict(CostAssumptions())])
    assumptions.to_json(paths.external_data_dir / "default_cost_assumptions.json", orient="records", indent=2)

    return {"sales": sales, "yield": yield_df, "costs": costs}
