import pandas as pd

from aged_cheddar_forecasting.decision import optimize_procurement, scenario_cost


def test_scenario_cost_components_sum():
    cost = scenario_cost(
        milk_liters=10000,
        demand_kg=900,
        yield_rate=0.1,
        milk_cost_per_liter=0.6,
        holding_cost_per_kg=0.2,
        stockout_cost_per_kg=2.0,
        spoilage_cost_per_kg=0.5,
        spoilage_rate=0.2,
    )
    assert round(cost.total, 6) == round(cost.milk_cost + cost.holding_cost + cost.stockout_cost + cost.spoilage_cost, 6)


def test_optimize_procurement_returns_expected_columns():
    cost_row = pd.Series(
        {
            "milk_cost_per_liter": 0.63,
            "holding_cost_per_kg": 0.18,
            "stockout_cost_per_kg": 2.4,
            "spoilage_cost_per_kg": 0.55,
            "spoilage_rate": 0.2,
        }
    )

    grid, best = optimize_procurement(
        demand_quantiles={"q10": 850, "q50": 1000, "q90": 1150},
        yield_quantiles={"q10": 0.095, "q50": 0.102, "q90": 0.109},
        cost_row=cost_row,
        n_samples=300,
    )

    assert not grid.empty
    assert "expected_total_cost" in grid.columns
    assert best["expected_total_cost"] == grid["expected_total_cost"].min()
