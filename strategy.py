from __future__ import annotations

import numpy as np
import pandas as pd

from backtest import MarketBundle


def generate_position(features: pd.DataFrame, market: MarketBundle) -> pd.Series:
    momentum_signal = np.tanh(features["normalized_momentum"] / 8.0)
    trend_signal = np.tanh(features["trend_strength"] * 40.0)
    raw_signal = 0.7 * momentum_signal + 0.3 * trend_signal

    volatility_scale = 1.0 / (1.0 + features["volatility_72h"] * 60.0)
    scaled_signal = raw_signal * volatility_scale

    positive_funding = features["funding_latest"].clip(lower=0.0)
    negative_funding = (-features["funding_latest"].clip(upper=0.0))

    long_penalty = np.clip(1.0 - positive_funding * 1200.0, 0.0, 1.0)
    short_penalty = np.clip(1.0 - negative_funding * 1200.0, 0.0, 1.0)

    adjusted = scaled_signal.copy()
    adjusted = adjusted.where(adjusted <= 0.0, adjusted * long_penalty)
    adjusted = adjusted.where(adjusted >= 0.0, adjusted * short_penalty)

    positions = adjusted.clip(-1.0, 1.0).fillna(0.0)
    return pd.Series(positions, index=features.index, name="position")

