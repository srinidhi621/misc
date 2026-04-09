from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CostAssumptions:
    milk_cost_per_liter: float = 0.63
    holding_cost_per_kg: float = 0.18
    stockout_cost_per_kg: float = 2.40
    spoilage_cost_per_kg: float = 0.55
    spoilage_rate: float = 0.20


@dataclass(frozen=True)
class Paths:
    project_root: Path = Path(__file__).resolve().parents[2]
    raw_data_dir: Path = project_root / "data" / "raw"
    external_data_dir: Path = project_root / "data" / "external"


RANDOM_SEED = 42
FORECAST_HORIZON_WEEKS = 26
