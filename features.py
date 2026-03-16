from __future__ import annotations

import numpy as np
import pandas as pd

from backtest import MarketBundle


def build_features(market: MarketBundle) -> pd.DataFrame:
    frame = market.frame
    close = frame["close"]
    returns_1h = close.pct_change().fillna(0.0)
    ema_bear_fast = close.ewm(span=24 * 7, adjust=False).mean()
    ema_bear_slow = close.ewm(span=24 * 30, adjust=False).mean()
    ema_fast = close.ewm(span=24 * 14, adjust=False).mean()
    ema_slow = close.ewm(span=24 * 60, adjust=False).mean()
    trend_regime_7d = (ema_bear_fast / ema_bear_slow - 1.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    trend_regime = (ema_fast / ema_slow - 1.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    trend_slope = ema_fast.pct_change(24 * 3).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    volatility_14d = returns_1h.rolling(24 * 14, min_periods=24 * 3).std().fillna(0.0)
    funding_latest = frame["funding_rate"].replace(0.0, np.nan).ffill().fillna(0.0)
    basis = (frame["mark_close"] / frame["index_close"] - 1.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    premium_7d = frame["premium_close"].rolling(24 * 7, min_periods=24).mean().fillna(0.0)
    breakout_20d = (
        close / close.rolling(24 * 20, min_periods=24 * 5).max() - 1.0
    ).replace([np.inf, -np.inf], 0.0).fillna(-1.0)
    return_3d = close.pct_change(24 * 3).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    drawdown_7d = (
        close / close.rolling(24 * 7, min_periods=24 * 3).max() - 1.0
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    features = pd.DataFrame(
        {
            "returns_1h": returns_1h,
            "trend_regime_7d": trend_regime_7d,
            "trend_regime": trend_regime,
            "trend_slope": trend_slope,
            "volatility_14d": volatility_14d,
            "funding_latest": funding_latest,
            "basis": basis,
            "premium_7d": premium_7d,
            "breakout_20d": breakout_20d,
            "return_3d": return_3d,
            "drawdown_7d": drawdown_7d,
        },
        index=frame.index,
    )
    return features.replace([np.inf, -np.inf], 0.0).fillna(0.0)
