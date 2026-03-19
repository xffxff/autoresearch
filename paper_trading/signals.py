from __future__ import annotations

from typing import Callable

import pandas as pd

from backtest import MarketBundle, validate_features, validate_positions


FeatureBuilder = Callable[[MarketBundle], pd.DataFrame]
StrategyBuilder = Callable[[pd.DataFrame, MarketBundle], pd.Series]


def make_market_bundle(frame: pd.DataFrame, symbol: str, interval: str) -> MarketBundle:
    return MarketBundle(frame=frame, symbol=symbol, bar_interval=interval)


def compute_signal(
    frame: pd.DataFrame,
    feature_builder: FeatureBuilder,
    strategy_builder: StrategyBuilder,
    symbol: str,
    interval: str,
) -> tuple[pd.DataFrame, pd.Series]:
    market = make_market_bundle(frame, symbol=symbol, interval=interval)
    features = feature_builder(market)
    validate_features(features, market)
    positions = strategy_builder(features, market)
    validate_positions(positions, market)
    return features, positions


def latest_target_position(
    frame: pd.DataFrame,
    feature_builder: FeatureBuilder,
    strategy_builder: StrategyBuilder,
    symbol: str,
    interval: str,
) -> float:
    _, positions = compute_signal(
        frame,
        feature_builder=feature_builder,
        strategy_builder=strategy_builder,
        symbol=symbol,
        interval=interval,
    )
    return float(positions.iloc[-1])
