from __future__ import annotations

import numpy as np
import pandas as pd

from backtest import MarketBundle


def build_features(market: MarketBundle) -> pd.DataFrame:
    frame = market.frame
    close = frame["close"]
    returns_1h = close.pct_change().fillna(0.0)
    momentum_24h = close.pct_change(24).fillna(0.0)
    volatility_72h = returns_1h.rolling(72, min_periods=24).std().fillna(0.0)
    normalized_momentum = momentum_24h / volatility_72h.clip(lower=1e-6)
    funding_latest = frame["funding_rate"].replace(0.0, np.nan).ffill().fillna(0.0)
    funding_24h = funding_latest.rolling(24, min_periods=1).mean().fillna(0.0)
    basis = (frame["mark_close"] / frame["index_close"] - 1.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    premium_24h = frame["premium_close"].rolling(24, min_periods=1).mean().fillna(0.0)
    fast_trend = close.ewm(span=24, adjust=False).mean()
    slow_trend = close.ewm(span=72, adjust=False).mean()
    trend_strength = (fast_trend / slow_trend - 1.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    features = pd.DataFrame(
        {
            "returns_1h": returns_1h,
            "momentum_24h": momentum_24h,
            "volatility_72h": volatility_72h,
            "normalized_momentum": normalized_momentum,
            "funding_latest": funding_latest,
            "funding_24h": funding_24h,
            "basis": basis,
            "premium_24h": premium_24h,
            "trend_strength": trend_strength,
        },
        index=frame.index,
    )
    return features.replace([np.inf, -np.inf], 0.0).fillna(0.0)

