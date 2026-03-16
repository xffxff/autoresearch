from __future__ import annotations

import pandas as pd

from backtest import MarketBundle


def generate_position(features: pd.DataFrame, market: MarketBundle) -> pd.Series:
    positions: list[float] = []
    cooldown_hours = 73
    funding_cap = 0.0006
    partial_premium_floor = -0.0003
    partial_size = 0.25
    stretched_trend_cap = 0.20
    volatility_cap = 0.006
    continuation_regime_floor = 0.01
    continuation_breakout_floor = -0.005
    state = 0.0
    hours_since_change = cooldown_hours

    for _, row in features.iterrows():
        hours_since_change += 1
        target = 0.0

        slow_trend_precheck = (
            row["trend_regime"] >= 0.009
            and row["trend_slope"] > -0.005
            and row["funding_latest"] < funding_cap
        )
        slow_trend_ready = slow_trend_precheck and row["premium_7d"] >= -0.00015
        breakout_reentry = (
            row["breakout_20d"] > -0.012
            and row["trend_slope"] > -0.003
            and row["funding_latest"] < funding_cap
            and row["trend_regime_7d"] > 0.007
        )

        base_filters_ready = row["return_3d"] > -0.05 and row["drawdown_7d"] > -0.08
        dual_volatility_ready = row["volatility_14d"] < volatility_cap and row["volatility_30d"] < volatility_cap
        strong_continuation_ready = (
            row["trend_regime_7d"] > continuation_regime_floor
            and row["breakout_20d"] > continuation_breakout_floor
            and row["volatility_30d"] < volatility_cap
        )
        filters_ready = base_filters_ready and (dual_volatility_ready or strong_continuation_ready)
        if filters_ready:
            if slow_trend_ready:
                target = partial_size if row["trend_regime"] > stretched_trend_cap else 1.0
            elif breakout_reentry:
                target = 1.0
            elif slow_trend_precheck and row["premium_7d"] >= partial_premium_floor:
                target = partial_size

        if row["trend_slope"] < -0.01 or row["funding_latest"] > 0.0009:
            if target > 0.0:
                target = 0.0

        if state == 0.0:
            if target != 0.0 and hours_since_change >= cooldown_hours:
                state = target
                hours_since_change = 0
        else:
            if target == 0.0 and hours_since_change >= cooldown_hours:
                state = 0.0
                hours_since_change = 0

        positions.append(state)

    return pd.Series(positions, index=features.index, name="position")
