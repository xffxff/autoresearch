from __future__ import annotations

import pandas as pd

from strategy import describe_signal, generate_position


def make_features() -> pd.DataFrame:
    index = pd.date_range("2026-03-18 00:00:00", periods=2, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "trend_regime_7d": [0.015, 0.002],
            "trend_regime": [0.02, -0.03],
            "trend_slope": [0.002, 0.001],
            "volatility_14d": [0.004, 0.004],
            "volatility_30d": [0.005, 0.005],
            "funding_latest": [0.0001, 0.0001],
            "premium_7d": [-0.0001, -0.0004],
            "breakout_20d": [-0.002, -0.08],
        },
        index=index,
    )


def test_describe_signal_matches_position_state_machine() -> None:
    features = make_features()
    market = type("Bundle", (), {"frame": features})()

    diagnostics = describe_signal(features, market)
    positions = generate_position(features, market)

    assert list(diagnostics["effective_position"]) == list(positions)
    assert diagnostics.iloc[0]["transition"] == "enter_long"
    assert bool(diagnostics.iloc[0]["slow_trend_ready"]) is True
    assert diagnostics.iloc[0]["signal_path"] == "slow_trend_full"
    assert diagnostics.iloc[1]["transition"] == "exit_cooldown"
    assert diagnostics.iloc[1]["effective_position"] == 1.0
