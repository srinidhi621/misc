from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ScenarioCost:
    milk_cost: float
    holding_cost: float
    stockout_cost: float
    spoilage_cost: float

    @property
    def total(self) -> float:
        return self.milk_cost + self.holding_cost + self.stockout_cost + self.spoilage_cost


def quantiles_to_normal_params(q10: float, q50: float, q90: float) -> tuple[float, float]:
    sigma = max((q90 - q10) / (2 * 1.2815515655446004), 1e-6)
    mu = q50
    return mu, sigma


def simulate_joint_outcomes(
    demand_quantiles: dict[str, float],
    yield_quantiles: dict[str, float],
    n_samples: int = 3000,
    correlation: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    corr = np.array([[1.0, correlation], [correlation, 1.0]])
    z = rng.multivariate_normal(mean=[0.0, 0.0], cov=corr, size=n_samples)

    d_mu, d_sigma = quantiles_to_normal_params(demand_quantiles["q10"], demand_quantiles["q50"], demand_quantiles["q90"])
    y_mu, y_sigma = quantiles_to_normal_params(yield_quantiles["q10"], yield_quantiles["q50"], yield_quantiles["q90"])

    demand = np.clip(d_mu + d_sigma * z[:, 0], 0, None)
    yield_rate = np.clip(y_mu + y_sigma * z[:, 1], 0.07, 0.14)

    return pd.DataFrame({"demand_kg": demand, "yield_rate": yield_rate})


def scenario_cost(
    milk_liters: float,
    demand_kg: float,
    yield_rate: float,
    milk_cost_per_liter: float,
    holding_cost_per_kg: float,
    stockout_cost_per_kg: float,
    spoilage_cost_per_kg: float,
    spoilage_rate: float,
) -> ScenarioCost:
    produced_kg = milk_liters * yield_rate
    surplus = max(produced_kg - demand_kg, 0.0)
    unmet = max(demand_kg - produced_kg, 0.0)

    return ScenarioCost(
        milk_cost=milk_liters * milk_cost_per_liter,
        holding_cost=surplus * holding_cost_per_kg,
        stockout_cost=unmet * stockout_cost_per_kg,
        spoilage_cost=(surplus * spoilage_rate) * spoilage_cost_per_kg,
    )


def optimize_procurement(
    demand_quantiles: dict[str, float],
    yield_quantiles: dict[str, float],
    cost_row: pd.Series,
    n_samples: int = 3000,
) -> tuple[pd.DataFrame, pd.Series]:
    sims = simulate_joint_outcomes(demand_quantiles, yield_quantiles, n_samples=n_samples)
    median_yield = max(yield_quantiles["q50"], 1e-6)
    implied_liters = demand_quantiles["q50"] / median_yield

    candidate_liters = np.linspace(0.8 * implied_liters, 1.25 * implied_liters, 35)

    rows = []
    for liters in candidate_liters:
        costs = sims.apply(
            lambda r: scenario_cost(
                milk_liters=float(liters),
                demand_kg=float(r["demand_kg"]),
                yield_rate=float(r["yield_rate"]),
                milk_cost_per_liter=float(cost_row["milk_cost_per_liter"]),
                holding_cost_per_kg=float(cost_row["holding_cost_per_kg"]),
                stockout_cost_per_kg=float(cost_row["stockout_cost_per_kg"]),
                spoilage_cost_per_kg=float(cost_row["spoilage_cost_per_kg"]),
                spoilage_rate=float(cost_row["spoilage_rate"]),
            ),
            axis=1,
        )

        milk = np.mean([c.milk_cost for c in costs])
        holding = np.mean([c.holding_cost for c in costs])
        stockout = np.mean([c.stockout_cost for c in costs])
        spoilage = np.mean([c.spoilage_cost for c in costs])

        rows.append(
            {
                "milk_liters": liters,
                "expected_milk_cost": milk,
                "expected_holding_cost": holding,
                "expected_stockout_cost": stockout,
                "expected_spoilage_cost": spoilage,
                "expected_total_cost": milk + holding + stockout + spoilage,
            }
        )

    summary = pd.DataFrame(rows)
    optimal = summary.loc[summary["expected_total_cost"].idxmin()]
    return summary, optimal
