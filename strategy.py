from __future__ import annotations

import numpy as np
import pandas as pd

from backtest import MarketBundle


def generate_position(features: pd.DataFrame, market: MarketBundle) -> pd.Series:
    positions: list[float] = []
    state = 0.0
    hours_since_change = 24
    hours_since_rebalance = 24

    for _, row in features.iterrows():
        hours_since_change += 1
        hours_since_rebalance += 1
        target = 0.0

        slow_trend_ready = (
            row["trend_regime"] >= 0.009
            and row["trend_slope"] > -0.005
            and row["funding_latest"] < 0.00065
            and row["premium_7d"] >= -0.00015
        )
        breakout_reentry = (
            row["breakout_20d"] > -0.01
            and row["trend_slope"] > -0.003
            and row["funding_latest"] < 0.00055
            and row["trend_regime_7d"] > 0.0
        )
        short_overlay_ready = (
            row["trend_regime"] <= -0.03
            and row["trend_regime_7d"] <= -0.015
            and row["trend_slope"] < 0.002
            and row["premium_7d"] <= -0.00035
            and row["funding_latest"] > -0.00025
        )

        if slow_trend_ready or breakout_reentry:
            target = 1.0
            vol_target = min(1.0, max(0.35, 0.22 / max(row["volatility_14d"], 0.003)))
            target = min(target, vol_target)
        elif short_overlay_ready:
            target = -0.20
            if row["trend_regime_7d"] > -0.03:
                target = -0.08
            elif row["trend_regime"] <= -0.07 or row["breakout_20d"] <= -0.12:
                target = -0.25

            vol_target = min(0.7, max(0.15, 0.18 / max(row["volatility_14d"], 0.003)))
            target = -min(abs(target), vol_target)

        if (
            row["trend_regime"] < 0.006
            or row["trend_slope"] < -0.013
            or row["funding_latest"] > 0.0009
        ):
            if target > 0.0:
                target = 0.0
        if (
            row["trend_regime"] > -0.002
            or row["trend_slope"] > 0.012
            or row["funding_latest"] < -0.00045
        ):
            if target < 0.0:
                target = 0.0

        if state == 0.0:
            if target != 0.0 and hours_since_change >= 24:
                state = target
                hours_since_change = 0
                hours_since_rebalance = 0
        else:
            if target == 0.0 and hours_since_change >= 24:
                state = 0.0
                hours_since_change = 0
                hours_since_rebalance = 0
            elif target != 0.0 and np.sign(target) != np.sign(state) and hours_since_change >= 24:
                state = target
                hours_since_change = 0
                hours_since_rebalance = 0
            elif target != 0.0 and abs(target - state) > 0.05 and hours_since_rebalance >= 24:
                state = target
                hours_since_rebalance = 0

        positions.append(state)

    return pd.Series(positions, index=features.index, name="position")
