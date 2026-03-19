from __future__ import annotations

from typing import Any

import pandas as pd

from paper_trading.market_data import (
    HyperliquidMarketData,
    normalize_candles,
    normalize_funding,
)


class StubInfoClient:
    def __init__(self, candles: list[dict[str, Any]]):
        self._candles = candles

    def candles_snapshot(
        self, symbol: str, interval: str, start_time: int, end_time: int
    ) -> list[dict[str, Any]]:
        assert symbol == "BTC"
        assert interval == "1h"
        return self._candles


def test_normalize_candles_uses_exchange_close_time_to_flag_incomplete_rows() -> None:
    now = pd.Timestamp("2026-03-18T13:08:02+00:00")
    raw = [
        {
            "t": 1773835200000,
            "T": 1773838799999,
            "o": "72915.0",
            "h": "72935.0",
            "l": "72244.0",
            "c": "72332.0",
            "v": "2169.9905",
        },
        {
            "t": 1773838800000,
            "T": 1773842399999,
            "o": "72332.0",
            "h": "72404.0",
            "l": "72075.0",
            "c": "72336.0",
            "v": "534.02723",
        },
    ]

    closed_only = normalize_candles(
        raw, interval="1h", now=now, include_incomplete=False
    )
    with_incomplete = normalize_candles(
        raw, interval="1h", now=now, include_incomplete=True
    )

    assert len(closed_only) == 1
    assert len(with_incomplete) == 2
    assert bool(with_incomplete.iloc[0]["is_closed"]) is True
    assert bool(with_incomplete.iloc[1]["is_closed"]) is False


def test_normalize_funding_floors_events_to_hour_and_keeps_latest() -> None:
    raw = [
        {
            "time": 1710115200008,
            "fundingRate": "0.0000790364",
            "premium": "0.0011322912",
        },
        {
            "time": 1710115200999,
            "fundingRate": "0.0000100000",
            "premium": "0.0020000000",
        },
    ]

    frame = normalize_funding(raw)

    assert len(frame) == 1
    assert frame.iloc[0]["funding_rate"] == 0.00001
    assert frame.iloc[0]["premium_close"] == 0.002


def test_bootstrap_candles_respects_max_backfill_limit() -> None:
    raw = [
        {
            "t": 1773835200000,
            "T": 1773838799999,
            "o": "72915.0",
            "h": "72935.0",
            "l": "72244.0",
            "c": "72332.0",
            "v": "2169.9905",
        }
    ]
    market_data = HyperliquidMarketData(
        symbol="BTC",
        interval="1h",
        base_url="https://api.hyperliquid.xyz",
        info_client=StubInfoClient(raw),
    )
    frame = market_data.bootstrap_candles(
        100,
        max_backfill_bars=1,
        now=pd.Timestamp("2026-03-18T13:08:02+00:00"),
    )

    assert len(frame) == 1
