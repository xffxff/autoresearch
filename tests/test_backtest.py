from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest import (
    EvaluationConfig,
    MarketBundle,
    evaluate_strategy,
    make_walk_forward_windows,
    simulate_validation_window,
)


def make_market_frame(length: int, drift: float = 0.0002) -> pd.DataFrame:
    index = pd.date_range("2021-01-01", periods=length, freq="1h", tz="UTC")
    open_ = 10_000 * np.cumprod(np.full(length, 1.0 + drift))
    frame = pd.DataFrame(
        {
            "open": open_,
            "high": open_ * 1.001,
            "low": open_ * 0.999,
            "close": open_ * (1.0 + drift / 2.0),
            "volume": np.full(length, 100.0),
            "mark_close": open_ * (1.0 + drift / 2.0),
            "index_close": open_ * (1.0 + drift / 3.0),
            "premium_close": np.zeros(length),
            "funding_rate": np.zeros(length),
        },
        index=index,
    )
    return frame


def test_simulate_validation_window_applies_lag_costs_and_funding() -> None:
    index = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [100.0, 100.0, 110.0, 121.0, 121.0],
            "high": [100.0, 110.0, 121.0, 121.0, 121.0],
            "low": [100.0, 100.0, 110.0, 121.0, 121.0],
            "close": [100.0, 110.0, 121.0, 121.0, 121.0],
            "volume": [1.0] * 5,
            "mark_close": [100.0, 110.0, 121.0, 121.0, 121.0],
            "index_close": [100.0, 110.0, 121.0, 121.0, 121.0],
            "premium_close": [0.0] * 5,
            "funding_rate": [0.0, 0.0, 0.01, 0.0, 0.0],
        },
        index=index,
    )
    market = MarketBundle(
        frame=frame,
        calibration_start=index[0],
        calibration_end=index[1],
        validation_start=index[1],
        validation_end=index[-1] + pd.Timedelta(hours=1),
    )
    positions = pd.Series([0.0, 1.0, 1.0, 1.0, 1.0], index=index)
    config = EvaluationConfig(taker_fee_bps=0.0, slippage_bps=0.0)

    result = simulate_validation_window(market, positions, config)

    expected_returns = pd.Series([0.0, 0.09, 0.0], index=index[1:4], dtype=float)
    pd.testing.assert_series_equal(
        result.returns.reset_index(drop=True),
        expected_returns.reset_index(drop=True),
        check_names=False,
    )
    assert result.trade_count == 1
    assert result.net_return == pytest.approx(0.09)


def test_make_walk_forward_windows_builds_six_sequential_windows() -> None:
    frame = make_market_frame(length=24 * 950)
    windows = make_walk_forward_windows(frame.index, EvaluationConfig())

    assert len(windows) == 6
    assert windows[0].validation_start - windows[0].calibration_start == pd.Timedelta(days=365)
    assert windows[1].validation_start - windows[0].validation_start == pd.Timedelta(days=90)


def test_evaluate_strategy_rejects_nan_features() -> None:
    frame = make_market_frame(length=24 * 950)
    market = MarketBundle(frame=frame)

    def bad_features(window_market: MarketBundle) -> pd.DataFrame:
        return pd.DataFrame({"x": np.nan}, index=window_market.frame.index)

    def good_strategy(features: pd.DataFrame, window_market: MarketBundle) -> pd.Series:
        return pd.Series(0.0, index=features.index)

    with pytest.raises(ValueError, match="must not contain NaN"):
        evaluate_strategy(market, bad_features, good_strategy)
