from __future__ import annotations

import pandas as pd

from backtest import MarketBundle


COOLDOWN_HOURS = 73
FUNDING_CAP = 0.0006
PARTIAL_PREMIUM_FLOOR = -0.0003
FULL_PREMIUM_FLOOR = -0.00015
PARTIAL_SIZE = 0.25
STRETCHED_TREND_CAP = 0.20
VOLATILITY_CAP = 0.006
CONTINUATION_REGIME_FLOOR = 0.01
CONTINUATION_BREAKOUT_FLOOR = -0.005
ENTRY_TREND_REGIME_FLOOR = 0.009
ENTRY_TREND_SLOPE_FLOOR = -0.005
REENTRY_TREND_SLOPE_FLOOR = -0.003
REENTRY_BREAKOUT_FLOOR = -0.012
REENTRY_TREND_REGIME_7D_FLOOR = 0.007
HARD_BLOCK_TREND_SLOPE = -0.01
HARD_BLOCK_FUNDING = 0.0009


def _signal_snapshot(row: pd.Series) -> dict[str, object]:
    slow_trend_precheck = (
        row["trend_regime"] >= ENTRY_TREND_REGIME_FLOOR
        and row["trend_slope"] > ENTRY_TREND_SLOPE_FLOOR
        and row["funding_latest"] < FUNDING_CAP
    )
    slow_trend_ready = slow_trend_precheck and row["premium_7d"] >= FULL_PREMIUM_FLOOR
    breakout_reentry = (
        row["breakout_20d"] > REENTRY_BREAKOUT_FLOOR
        and row["trend_slope"] > REENTRY_TREND_SLOPE_FLOOR
        and row["funding_latest"] < FUNDING_CAP
        and row["trend_regime_7d"] > REENTRY_TREND_REGIME_7D_FLOOR
    )
    dual_volatility_ready = (
        row["volatility_14d"] < VOLATILITY_CAP
        and row["volatility_30d"] < VOLATILITY_CAP
    )
    strong_continuation_ready = (
        row["trend_regime_7d"] > CONTINUATION_REGIME_FLOOR
        and row["breakout_20d"] > CONTINUATION_BREAKOUT_FLOOR
        and row["volatility_30d"] < VOLATILITY_CAP
    )
    volatility_ready = dual_volatility_ready or strong_continuation_ready
    partial_ready = slow_trend_precheck and row["premium_7d"] >= PARTIAL_PREMIUM_FLOOR

    candidate_target = 0.0
    signal_path = "flat_no_entry"
    if volatility_ready:
        if slow_trend_ready:
            if row["trend_regime"] > STRETCHED_TREND_CAP:
                candidate_target = PARTIAL_SIZE
                signal_path = "slow_trend_stretched_partial"
            else:
                candidate_target = 1.0
                signal_path = "slow_trend_full"
        elif breakout_reentry:
            candidate_target = 1.0
            signal_path = "breakout_reentry"
        elif partial_ready:
            candidate_target = PARTIAL_SIZE
            signal_path = "slow_trend_partial"
        else:
            signal_path = "volatility_ready_no_entry"
    else:
        signal_path = "volatility_blocked"

    hard_block = (
        row["trend_slope"] < HARD_BLOCK_TREND_SLOPE
        or row["funding_latest"] > HARD_BLOCK_FUNDING
    )
    signal_target = candidate_target
    if hard_block and candidate_target > 0.0:
        candidate_target = 0.0
        signal_path = "hard_blocked"

    return {
        "slow_trend_precheck": slow_trend_precheck,
        "slow_trend_ready": slow_trend_ready,
        "breakout_reentry": breakout_reentry,
        "dual_volatility_ready": dual_volatility_ready,
        "strong_continuation_ready": strong_continuation_ready,
        "volatility_ready": volatility_ready,
        "partial_ready": partial_ready,
        "hard_block": hard_block,
        "signal_target": float(signal_target),
        "candidate_target": float(candidate_target),
        "signal_path": signal_path,
    }


def describe_signal(features: pd.DataFrame, market: MarketBundle) -> pd.DataFrame:
    del market
    rows: list[dict[str, object]] = []
    state = 0.0
    hours_since_change = COOLDOWN_HOURS

    for timestamp, feature_row in features.iterrows():
        hours_since_change += 1
        snapshot = _signal_snapshot(feature_row)
        state_before = state
        candidate_target = float(snapshot["candidate_target"])
        cooldown_ready = hours_since_change >= COOLDOWN_HOURS
        cooldown_hours_remaining = max(COOLDOWN_HOURS - hours_since_change, 0)
        transition = "hold_flat"
        trade_intent = abs(candidate_target - state_before) > 1e-12

        if state == 0.0:
            if candidate_target != 0.0 and cooldown_ready:
                state = candidate_target
                hours_since_change = 0
                transition = "enter_long"
            elif candidate_target != 0.0:
                transition = "entry_cooldown"
        else:
            if candidate_target == 0.0 and cooldown_ready:
                state = 0.0
                hours_since_change = 0
                transition = "exit_long"
            elif candidate_target == 0.0:
                transition = "exit_cooldown"
            else:
                transition = "hold_long"

        rows.append(
            {
                **feature_row.to_dict(),
                **snapshot,
                "bar_close_time": timestamp.isoformat(),
                "state_before": float(state_before),
                "effective_position": float(state),
                "cooldown_ready": cooldown_ready,
                "cooldown_hours_remaining": int(cooldown_hours_remaining),
                "trade_intent": trade_intent,
                "transition": transition,
            }
        )

    return pd.DataFrame(rows, index=features.index)


def generate_position(features: pd.DataFrame, market: MarketBundle) -> pd.Series:
    diagnostics = describe_signal(features, market)
    return pd.Series(
        diagnostics["effective_position"], index=features.index, name="position"
    )
