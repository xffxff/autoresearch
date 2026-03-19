from __future__ import annotations

import pandas as pd


REQUIRED_PAPER_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "mark_close",
    "index_close",
    "premium_close",
    "funding_rate",
]


def build_market_frame(
    candles: pd.DataFrame, funding: pd.DataFrame | None = None
) -> pd.DataFrame:
    if candles.empty:
        raise ValueError("paper candles frame must not be empty")
    frame = candles[["open", "high", "low", "close", "volume"]].copy()
    frame["mark_close"] = frame["close"]
    frame["index_close"] = frame["close"]
    frame["premium_close"] = 0.0
    frame["funding_rate"] = 0.0
    if funding is not None and not funding.empty:
        normalized = funding.copy().sort_index()
        if (
            "premium_close" not in normalized.columns
            and "premium" in normalized.columns
        ):
            normalized["premium_close"] = normalized["premium"]
        if "funding_rate" in normalized.columns:
            frame["funding_rate"] = (
                normalized["funding_rate"].reindex(frame.index).fillna(0.0)
            )
        if "premium_close" in normalized.columns:
            premium_series = normalized["premium_close"].reindex(frame.index)
            frame["premium_close"] = premium_series.ffill().fillna(0.0)
    frame = frame.sort_index()
    return frame[REQUIRED_PAPER_COLUMNS].astype(float)


def ensure_warmup(frame: pd.DataFrame, warmup_bars: int) -> None:
    if len(frame) < warmup_bars:
        raise ValueError(
            f"paper warmup requires at least {warmup_bars} closed bars, got {len(frame)}"
        )
