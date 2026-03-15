from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest import (
    EvaluationConfig,
    MarketBundle,
    WindowResult,
    annualized_sharpe,
    cumulative_return,
    evaluate_strategy,
    max_drawdown,
    make_holdout_window,
    make_walk_forward_windows,
    simulate_validation_window,
    summarize_window_results,
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


def make_window_result(
    start: str,
    returns: list[float],
    *,
    trade_count: int = 10,
    turnover: float = 20.0,
) -> WindowResult:
    index = pd.date_range(start, periods=len(returns), freq="1h", tz="UTC")
    series = pd.Series(returns, index=index, dtype=float)
    return WindowResult(
        calibration_start=index[0] - pd.Timedelta(days=365),
        validation_start=index[0],
        validation_end=index[-1] + pd.Timedelta(hours=1),
        net_sharpe=annualized_sharpe(series),
        net_return=cumulative_return(series),
        max_drawdown=max_drawdown(series),
        trade_count=trade_count,
        turnover=turnover,
        bars=len(series),
        returns=series,
    )


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
    assert result.benchmark_net_return == pytest.approx(0.21)
    assert result.benchmark_max_drawdown == pytest.approx(0.0)


def test_make_walk_forward_windows_builds_six_sequential_windows() -> None:
    frame = make_market_frame(length=24 * 1_200)
    windows = make_walk_forward_windows(frame.index, EvaluationConfig())

    assert len(windows) == 6
    assert windows[0].validation_start - windows[0].calibration_start == pd.Timedelta(days=365)
    assert windows[1].validation_start - windows[0].validation_start == pd.Timedelta(days=90)


def test_make_holdout_window_reserves_final_ninety_days() -> None:
    frame = make_market_frame(length=24 * 1_200)
    config = EvaluationConfig()
    windows = make_walk_forward_windows(frame.index, config)
    holdout = make_holdout_window(frame.index, config)

    assert holdout is not None
    assert holdout.validation_start == windows[-1].validation_end
    assert holdout.validation_end - holdout.validation_start == pd.Timedelta(days=config.holdout_days)


def test_evaluate_strategy_rejects_nan_features() -> None:
    frame = make_market_frame(length=24 * 1_200)
    market = MarketBundle(frame=frame)

    def bad_features(window_market: MarketBundle) -> pd.DataFrame:
        return pd.DataFrame({"x": np.nan}, index=window_market.frame.index)

    def good_strategy(features: pd.DataFrame, window_market: MarketBundle) -> pd.Series:
        return pd.Series(0.0, index=features.index)

    with pytest.raises(ValueError, match="must not contain NaN"):
        evaluate_strategy(market, bad_features, good_strategy)


def test_summarize_window_results_prefers_net_return_over_sharpe() -> None:
    high_return_windows = [
        make_window_result(f"2024-01-{day:02d}", [0.18, -0.14, 0.16, -0.10, 0.12], trade_count=8, turnover=25.0)
        for day in range(1, 7)
    ]
    high_sharpe_windows = [
        make_window_result(f"2024-02-{day:02d}", [0.003, 0.001] * 20, trade_count=8, turnover=25.0)
        for day in range(1, 7)
    ]
    loose_config = EvaluationConfig(
        min_trade_count=0,
        max_drawdown=1.0,
        max_annualized_turnover=1_000.0,
        min_active_windows=0,
        min_window_net_return=-1.0,
    )

    high_return_result = summarize_window_results(high_return_windows, loose_config)
    high_sharpe_result = summarize_window_results(high_sharpe_windows, loose_config)

    assert high_return_result.net_sharpe < high_sharpe_result.net_sharpe
    assert high_return_result.score > high_sharpe_result.score
    assert high_return_result.score == pytest.approx(high_return_result.net_return)


def test_summarize_window_results_gate_failures_are_independent() -> None:
    config = EvaluationConfig()
    valid_windows = [
        make_window_result(f"2024-03-{day:02d}", [0.02, -0.01, 0.03], trade_count=10, turnover=20.0)
        for day in range(1, 7)
    ]
    valid_result = summarize_window_results(valid_windows, config)
    assert valid_result.pass_gates is True

    low_trade_result = summarize_window_results(
        [
            make_window_result(f"2024-04-{day:02d}", [0.02, -0.01, 0.03], trade_count=trade_count, turnover=20.0)
            for day, trade_count in zip(range(1, 7), [5, 5, 3, 3, 2, 1], strict=True)
        ],
        config,
    )
    assert low_trade_result.trade_count == 19
    assert low_trade_result.pass_gates is False

    high_drawdown_windows = [make_window_result("2024-05-01", [2.0, -0.6], trade_count=10, turnover=20.0)]
    high_drawdown_windows.extend(
        make_window_result(f"2024-05-{day:02d}", [0.02, -0.01, 0.03], trade_count=10, turnover=20.0)
        for day in range(2, 7)
    )
    high_drawdown_result = summarize_window_results(high_drawdown_windows, config)
    assert high_drawdown_result.max_drawdown > config.max_drawdown
    assert high_drawdown_result.worst_window_return > config.min_window_net_return
    assert high_drawdown_result.pass_gates is False

    high_turnover_result = summarize_window_results(
        [
            make_window_result(f"2024-06-{day:02d}", [0.02, -0.01, 0.03], trade_count=10, turnover=200.0)
            for day in range(1, 7)
        ],
        config,
    )
    assert high_turnover_result.turnover > config.max_annualized_turnover
    assert high_turnover_result.pass_gates is False

    low_active_windows = [
        make_window_result("2024-07-01", [0.02, -0.01, 0.03], trade_count=15, turnover=20.0),
        make_window_result("2024-07-02", [0.02, -0.01, 0.03], trade_count=15, turnover=20.0),
        make_window_result("2024-07-03", [0.0, 0.0, 0.0], trade_count=0, turnover=0.0),
        make_window_result("2024-07-04", [0.0, 0.0, 0.0], trade_count=0, turnover=0.0),
        make_window_result("2024-07-05", [0.0, 0.0, 0.0], trade_count=0, turnover=0.0),
        make_window_result("2024-07-06", [0.0, 0.0, 0.0], trade_count=0, turnover=0.0),
    ]
    low_active_result = summarize_window_results(low_active_windows, config)
    assert low_active_result.trade_count >= config.min_trade_count
    assert low_active_result.active_windows == 2
    assert low_active_result.pass_gates is False

    bad_window_return_windows = [make_window_result("2024-08-01", [-0.25], trade_count=10, turnover=20.0)]
    bad_window_return_windows.extend(
        make_window_result(f"2024-08-{day:02d}", [0.04, 0.02], trade_count=10, turnover=20.0)
        for day in range(2, 7)
    )
    bad_window_return_result = summarize_window_results(bad_window_return_windows, config)
    assert bad_window_return_result.worst_window_return < config.min_window_net_return
    assert bad_window_return_result.pass_gates is False


def test_summarize_window_results_aggressive_gate_allows_high_return_deep_but_bounded_path() -> None:
    config = EvaluationConfig()
    windows = [
        make_window_result("2024-09-01", [-0.15], trade_count=8, turnover=25.0),
        make_window_result("2024-09-02", [1.0, -0.45], trade_count=8, turnover=25.0),
        make_window_result("2024-09-03", [0.06, 0.04], trade_count=8, turnover=25.0),
        make_window_result("2024-09-04", [0.05, 0.03], trade_count=8, turnover=25.0),
        make_window_result("2024-09-05", [0.04, 0.02], trade_count=8, turnover=25.0),
        make_window_result("2024-09-06", [0.03, 0.02], trade_count=8, turnover=25.0),
    ]

    result = summarize_window_results(windows, config)

    assert result.max_drawdown == pytest.approx(0.45, abs=1e-9)
    assert result.worst_window_return == pytest.approx(-0.15, abs=1e-9)
    assert result.net_return > 0.0
    assert result.pass_gates is True
