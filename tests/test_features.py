import pandas as pd

from aged_cheddar_forecasting.features import build_demand_features_asof


def test_demand_features_are_lagged_and_target_shifted():
    df = pd.DataFrame(
        {
            "week_start": pd.date_range("2024-01-01", periods=40, freq="W-MON"),
            "promo_intensity": 0,
            "competitor_price_index": 1.0,
            "holiday_flag": 0,
            "avg_temp_c": 10.0,
            "demand_kg": range(40),
        }
    )

    out = build_demand_features_asof(df, horizon_weeks=4)

    # For row index 10, lag_1 should reference demand at row 9, not current row 10.
    assert out.loc[10, "demand_kg_lag_1"] == 9
    # Demand drivers should also be lagged to keep strict as-of behavior.
    assert out.loc[10, "competitor_price_index_lag_1"] == 1.0
    # Target should be horizon weeks ahead.
    assert out.loc[10, "target_demand_kg"] == 14
    # Raw current-week columns are dropped to avoid accidental leakage.
    assert "demand_kg" not in out.columns
    assert "promo_intensity" not in out.columns
