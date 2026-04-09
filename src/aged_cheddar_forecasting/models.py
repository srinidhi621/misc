from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


@dataclass
class QuantileModelBundle:
    q10: GradientBoostingRegressor
    q50: GradientBoostingRegressor
    q90: GradientBoostingRegressor

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        p10 = self.q10.predict(x)
        p50 = self.q50.predict(x)
        p90 = self.q90.predict(x)
        return pd.DataFrame({"q10": p10, "q50": p50, "q90": p90}, index=x.index)


def train_quantile_bundle(x_train: pd.DataFrame, y_train: pd.Series, seed: int = 42) -> QuantileModelBundle:
    common = dict(n_estimators=250, learning_rate=0.05, max_depth=3, random_state=seed)

    q10 = GradientBoostingRegressor(loss="quantile", alpha=0.1, **common)
    q50 = GradientBoostingRegressor(loss="quantile", alpha=0.5, **common)
    q90 = GradientBoostingRegressor(loss="quantile", alpha=0.9, **common)

    q10.fit(x_train, y_train)
    q50.fit(x_train, y_train)
    q90.fit(x_train, y_train)

    return QuantileModelBundle(q10=q10, q50=q50, q90=q90)


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    err = y_true - y_pred
    return float(np.mean(np.maximum(q * err, (q - 1) * err)))
