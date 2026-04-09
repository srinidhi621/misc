from __future__ import annotations

from typing import Mapping


def _require_numeric(payload: Mapping[str, object], required: list[str]) -> None:
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    for key in required:
        value = payload[key]
        if not isinstance(value, (int, float)):
            raise TypeError(f"{key} must be numeric")


def generate_procurement_memo(summary: Mapping[str, object]) -> str:
    required = [
        "week_start",
        "recommended_milk_liters",
        "expected_total_cost",
        "expected_milk_cost",
        "expected_holding_cost",
        "expected_stockout_cost",
        "expected_spoilage_cost",
        "baseline_expected_total_cost",
        "estimated_savings_vs_baseline",
    ]
    _require_numeric(summary, [k for k in required if k != "week_start"])
    if "week_start" not in summary:
        raise ValueError("week_start is required")

    return (
        f"# Weekly Procurement Memo\n\n"
        f"## Decision\n"
        f"- Planning week start: **{summary['week_start']}**\n"
        f"- Recommended milk procurement: **{summary['recommended_milk_liters']:.0f} liters**\n\n"
        f"## Expected Cost Breakdown\n"
        f"- Milk cost: **${summary['expected_milk_cost']:.2f}**\n"
        f"- Holding cost: **${summary['expected_holding_cost']:.2f}**\n"
        f"- Stockout cost: **${summary['expected_stockout_cost']:.2f}**\n"
        f"- Spoilage cost: **${summary['expected_spoilage_cost']:.2f}**\n"
        f"- Total expected cost: **${summary['expected_total_cost']:.2f}**\n\n"
        f"## Baseline Comparison\n"
        f"- Baseline expected total cost: **${summary['baseline_expected_total_cost']:.2f}**\n"
        f"- Estimated savings vs baseline: **${summary['estimated_savings_vs_baseline']:.2f}**\n"
    )
