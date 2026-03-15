from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


HOURS_PER_YEAR = 24 * 365
REQUIRED_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "mark_close",
    "index_close",
    "premium_close",
    "funding_rate",
}


@dataclass(frozen=True)
class MarketBundle:
    frame: pd.DataFrame
    symbol: str = "BTCUSDT"
    bar_interval: str = "1h"
    calibration_start: pd.Timestamp | None = None
    calibration_end: pd.Timestamp | None = None
    validation_start: pd.Timestamp | None = None
    validation_end: pd.Timestamp | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationConfig:
    calibration_days: int = 365
    validation_days: int = 90
    num_windows: int = 6
    holdout_days: int = 90
    taker_fee_bps: float = 5.0
    slippage_bps: float = 1.0
    min_trade_count: int = 20
    max_drawdown: float = 0.55
    max_annualized_turnover: float = 150.0
    min_active_windows: int = 3
    min_window_net_return: float = -0.20


@dataclass(frozen=True)
class WindowSpec:
    calibration_start: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


@dataclass(frozen=True)
class WindowResult:
    calibration_start: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    net_sharpe: float
    net_return: float
    max_drawdown: float
    trade_count: int
    turnover: float
    bars: int
    returns: pd.Series
    benchmark_net_return: float = 0.0
    benchmark_max_drawdown: float = 0.0


@dataclass(frozen=True)
class BacktestResult:
    score: float
    net_sharpe: float
    net_return: float
    max_drawdown: float
    trade_count: int
    turnover: float
    active_windows: int
    worst_window_return: float
    best_window_return: float
    pass_gates: bool
    windows: list[WindowResult]
    holdout: WindowResult | None = None


def validate_market_frame(frame: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"market frame missing required columns: {sorted(missing)}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("market frame index must be a DatetimeIndex")
    if frame.index.tz is None:
        raise ValueError("market frame index must be timezone-aware")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("market frame index must be sorted ascending")
    if frame.index.has_duplicates:
        raise ValueError("market frame index must not contain duplicates")


def _analysis_end_and_base_start(
    index: pd.DatetimeIndex,
    config: EvaluationConfig,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if len(index) < 2:
        raise ValueError("dataset is too small for evaluation")
    analysis_end = index.max() + pd.Timedelta(hours=1)
    total_days = config.calibration_days + config.validation_days * config.num_windows + config.holdout_days
    base_start = analysis_end - pd.Timedelta(days=total_days)
    if index.min() > base_start:
        raise ValueError(
            f"dataset does not contain enough history for {config.num_windows} research windows "
            f"of {config.calibration_days}+{config.validation_days} days plus "
            f"{config.holdout_days} holdout days"
        )
    return analysis_end, base_start


def make_walk_forward_windows(index: pd.DatetimeIndex, config: EvaluationConfig) -> list[WindowSpec]:
    _, base_start = _analysis_end_and_base_start(index, config)

    windows: list[WindowSpec] = []
    for step in range(config.num_windows):
        calibration_start = base_start + pd.Timedelta(days=config.validation_days * step)
        validation_start = calibration_start + pd.Timedelta(days=config.calibration_days)
        validation_end = validation_start + pd.Timedelta(days=config.validation_days)
        windows.append(
            WindowSpec(
                calibration_start=calibration_start,
                validation_start=validation_start,
                validation_end=validation_end,
            )
        )
    return windows


def make_holdout_window(index: pd.DatetimeIndex, config: EvaluationConfig) -> WindowSpec | None:
    if config.holdout_days <= 0:
        return None
    _, base_start = _analysis_end_and_base_start(index, config)
    calibration_start = base_start + pd.Timedelta(days=config.validation_days * config.num_windows)
    validation_start = calibration_start + pd.Timedelta(days=config.calibration_days)
    validation_end = validation_start + pd.Timedelta(days=config.holdout_days)
    return WindowSpec(
        calibration_start=calibration_start,
        validation_start=validation_start,
        validation_end=validation_end,
    )


def validate_features(features: pd.DataFrame, market: MarketBundle) -> None:
    if not isinstance(features, pd.DataFrame):
        raise TypeError("build_features must return a pandas DataFrame")
    if features.empty:
        raise ValueError("features frame must not be empty")
    if not features.index.equals(market.frame.index):
        raise ValueError("features frame index must exactly match market.frame.index")
    if features.isnull().values.any():
        raise ValueError("features frame must not contain NaN values")
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("features frame must contain only finite values")


def validate_positions(positions: pd.Series, market: MarketBundle) -> None:
    if not isinstance(positions, pd.Series):
        raise TypeError("generate_position must return a pandas Series")
    if not positions.index.equals(market.frame.index):
        raise ValueError("position index must exactly match market.frame.index")
    if positions.isnull().any():
        raise ValueError("positions must not contain NaN values")
    if not np.isfinite(positions.to_numpy()).all():
        raise ValueError("positions must contain only finite values")
    if (positions.abs() > 1.0 + 1e-9).any():
        raise ValueError("positions must remain inside [-1, 1]")


def annualized_sharpe(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    volatility = returns.std(ddof=0)
    if volatility <= 1e-12:
        return 0.0
    return float(np.sqrt(HOURS_PER_YEAR) * returns.mean() / volatility)


def cumulative_return(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    return float((1.0 + returns).prod() - 1.0)


def max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(-drawdown.min())


def simulate_validation_window(
    market: MarketBundle,
    positions: pd.Series,
    config: EvaluationConfig,
) -> WindowResult:
    frame = market.frame.copy()
    frame["signal"] = positions
    frame["effective_position"] = frame["signal"].shift(1).fillna(0.0)
    frame["next_open_return"] = frame["open"].shift(-1) / frame["open"] - 1.0
    frame["delta_position"] = frame["effective_position"].diff().abs().fillna(frame["effective_position"].abs())

    per_side_cost = (config.taker_fee_bps + config.slippage_bps) / 10_000.0
    frame["trading_cost"] = frame["delta_position"] * per_side_cost
    frame["funding_pnl"] = -frame["effective_position"] * frame["funding_rate"].fillna(0.0)
    frame["net_return_component"] = (
        frame["effective_position"] * frame["next_open_return"]
        + frame["funding_pnl"]
        - frame["trading_cost"]
    )

    validation_mask = (frame.index >= market.validation_start) & (frame.index < market.validation_end)
    validation = frame.loc[validation_mask].copy()
    validation = validation[validation["next_open_return"].notna()].copy()
    if validation.empty:
        raise ValueError("validation slice is empty after applying the execution lag")

    returns = validation["net_return_component"]
    benchmark_returns = validation["next_open_return"]
    trade_count = int((validation["delta_position"] > 1e-12).sum())
    turnover = float(validation["delta_position"].sum() * HOURS_PER_YEAR / len(validation))

    return WindowResult(
        calibration_start=market.calibration_start,
        validation_start=market.validation_start,
        validation_end=market.validation_end,
        net_sharpe=annualized_sharpe(returns),
        net_return=cumulative_return(returns),
        max_drawdown=max_drawdown(returns),
        trade_count=trade_count,
        turnover=turnover,
        bars=len(validation),
        returns=returns,
        benchmark_net_return=cumulative_return(benchmark_returns),
        benchmark_max_drawdown=max_drawdown(benchmark_returns),
    )


def evaluate_strategy(
    market: MarketBundle,
    feature_builder: Callable[[MarketBundle], pd.DataFrame],
    strategy_builder: Callable[[pd.DataFrame, MarketBundle], pd.Series],
    config: EvaluationConfig | None = None,
    *,
    include_holdout: bool = False,
) -> BacktestResult:
    config = config or EvaluationConfig()
    validate_market_frame(market.frame)
    windows = make_walk_forward_windows(market.frame.index, config)

    window_results: list[WindowResult] = []
    for spec in windows:
        window_frame = market.frame.loc[(market.frame.index >= spec.calibration_start) & (market.frame.index < spec.validation_end)].copy()
        window_market = MarketBundle(
            frame=window_frame,
            symbol=market.symbol,
            bar_interval=market.bar_interval,
            calibration_start=spec.calibration_start,
            calibration_end=spec.validation_start,
            validation_start=spec.validation_start,
            validation_end=spec.validation_end,
        )
        features = feature_builder(window_market)
        validate_features(features, window_market)
        positions = strategy_builder(features, window_market)
        validate_positions(positions, window_market)
        window_results.append(simulate_validation_window(window_market, positions, config))

    holdout_result: WindowResult | None = None
    holdout_spec = make_holdout_window(market.frame.index, config) if include_holdout else None
    if holdout_spec is not None:
        holdout_frame = market.frame.loc[
            (market.frame.index >= holdout_spec.calibration_start)
            & (market.frame.index < holdout_spec.validation_end)
        ].copy()
        holdout_market = MarketBundle(
            frame=holdout_frame,
            symbol=market.symbol,
            bar_interval=market.bar_interval,
            calibration_start=holdout_spec.calibration_start,
            calibration_end=holdout_spec.validation_start,
            validation_start=holdout_spec.validation_start,
            validation_end=holdout_spec.validation_end,
        )
        holdout_features = feature_builder(holdout_market)
        validate_features(holdout_features, holdout_market)
        holdout_positions = strategy_builder(holdout_features, holdout_market)
        validate_positions(holdout_positions, holdout_market)
        holdout_result = simulate_validation_window(holdout_market, holdout_positions, config)

    return summarize_window_results(window_results, config, holdout_result=holdout_result)


def summarize_window_results(
    window_results: list[WindowResult],
    config: EvaluationConfig,
    holdout_result: WindowResult | None = None,
) -> BacktestResult:
    if not window_results:
        raise ValueError("window_results must not be empty")

    combined_returns = pd.concat([window.returns for window in window_results]).sort_index()
    weights = np.array([window.bars for window in window_results], dtype=float)
    total_net_return = cumulative_return(combined_returns)
    total_trade_count = int(sum(window.trade_count for window in window_results))
    total_turnover = float(np.average([window.turnover for window in window_results], weights=weights))
    total_max_drawdown = max_drawdown(combined_returns)
    active_windows = int(sum((window.trade_count > 0) or (abs(window.net_return) > 1e-12) for window in window_results))
    worst_window_return = float(min(window.net_return for window in window_results))
    best_window_return = float(max(window.net_return for window in window_results))

    pass_gates = (
        total_trade_count >= config.min_trade_count
        and total_max_drawdown <= config.max_drawdown
        and total_turnover <= config.max_annualized_turnover
        and active_windows >= config.min_active_windows
        and worst_window_return >= config.min_window_net_return
    )

    return BacktestResult(
        score=total_net_return,
        net_sharpe=annualized_sharpe(combined_returns),
        net_return=total_net_return,
        max_drawdown=total_max_drawdown,
        trade_count=total_trade_count,
        turnover=total_turnover,
        active_windows=active_windows,
        worst_window_return=worst_window_return,
        best_window_return=best_window_return,
        pass_gates=pass_gates,
        windows=window_results,
        holdout=holdout_result,
    )
