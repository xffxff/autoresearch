from __future__ import annotations

from typing import Any

import pandas as pd

from paper_trading.types import AssetContextSnapshot, BboSnapshot


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _interval_delta(interval: str) -> pd.Timedelta:
    if interval != "1h":
        raise ValueError("paper v1 only supports 1h bars")
    return pd.Timedelta(hours=1)


class HyperliquidMarketData:
    def __init__(
        self, symbol: str, interval: str, base_url: str, info_client: Any | None = None
    ):
        self.symbol = symbol
        self.interval = interval
        self.base_url = base_url
        self.info = info_client or self._build_info_client(base_url)

    @staticmethod
    def _build_info_client(base_url: str) -> Any:
        from hyperliquid.info import Info

        return Info(base_url, skip_ws=True)

    def bootstrap_candles(
        self,
        warmup_bars: int,
        *,
        max_backfill_bars: int = 5_000,
        include_incomplete: bool = False,
        now: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        limit = max(1, warmup_bars)
        if limit > max_backfill_bars:
            limit = max_backfill_bars
        now = now or _utc_now()
        interval_delta = _interval_delta(self.interval)
        start_time = now - interval_delta * limit
        raw = self.info.candles_snapshot(
            self.symbol,
            self.interval,
            int(start_time.timestamp() * 1000),
            int(now.timestamp() * 1000),
        )
        return normalize_candles(
            raw, interval=self.interval, now=now, include_incomplete=include_incomplete
        )

    def poll_recent_candles(
        self, lookback_bars: int = 3, *, now: pd.Timestamp | None = None
    ) -> pd.DataFrame:
        return self.bootstrap_candles(
            max(lookback_bars, 1),
            max_backfill_bars=max(lookback_bars, 1),
            include_incomplete=True,
            now=now,
        )

    def fetch_funding_history(
        self,
        start_time: pd.Timestamp,
        *,
        end_time: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        end_time = end_time or _utc_now()
        raw = self.info.funding_history(
            self.symbol,
            int(start_time.timestamp() * 1000),
            int(end_time.timestamp() * 1000),
        )
        return normalize_funding(raw)

    def fetch_bbo(self, *, now: pd.Timestamp | None = None) -> BboSnapshot | None:
        now = now or _utc_now()
        snapshot = self.info.l2_snapshot(self.symbol)
        levels = snapshot.get("levels", [])
        if len(levels) < 2 or not levels[0] or not levels[1]:
            return None
        bid_level = levels[0][0]
        ask_level = levels[1][0]
        return BboSnapshot(
            timestamp=now.isoformat(),
            bid_px=float(bid_level["px"]),
            ask_px=float(ask_level["px"]),
        )

    def fetch_asset_context(
        self, *, now: pd.Timestamp | None = None
    ) -> AssetContextSnapshot | None:
        now = now or _utc_now()
        meta, ctxs = self.info.meta_and_asset_ctxs()
        for asset_info, ctx in zip(meta["universe"], ctxs, strict=True):
            if asset_info["name"] != self.symbol:
                continue
            return AssetContextSnapshot(
                timestamp=now.isoformat(),
                mark_px=float(ctx["markPx"]),
                oracle_px=float(ctx["oraclePx"]),
                premium=float(ctx.get("premium") or 0.0),
                funding_rate=float(ctx.get("funding") or 0.0),
            )
        return None


def normalize_candles(
    raw_candles: list[dict[str, Any]],
    *,
    interval: str,
    now: pd.Timestamp | None = None,
    include_incomplete: bool = False,
) -> pd.DataFrame:
    now = now or _utc_now()
    interval_delta = _interval_delta(interval)
    records: list[dict[str, Any]] = []
    for item in raw_candles:
        open_time = pd.Timestamp(item["t"], unit="ms", tz="UTC")
        close_time = pd.Timestamp(
            item.get("T", item["t"]), unit="ms", tz="UTC"
        ) + pd.Timedelta(milliseconds=1)
        is_closed = close_time <= now
        if not include_incomplete and not is_closed:
            continue
        records.append(
            {
                "timestamp": open_time,
                "open": float(item["o"]),
                "high": float(item["h"]),
                "low": float(item["l"]),
                "close": float(item["c"]),
                "volume": float(item["v"]),
                "close_time": close_time,
                "is_closed": is_closed,
            }
        )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "is_closed",
            ]
        )
    return frame.set_index("timestamp").sort_index()


def normalize_funding(raw_funding: list[dict[str, Any]]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for item in raw_funding:
        timestamp = pd.Timestamp(item["time"], unit="ms", tz="UTC").floor("h")
        records.append(
            {
                "timestamp": timestamp,
                "funding_rate": float(item.get("fundingRate") or 0.0),
                "premium_close": float(item.get("premium") or 0.0),
            }
        )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return pd.DataFrame(columns=["funding_rate", "premium_close"])
    frame = (
        frame.drop_duplicates(subset="timestamp", keep="last")
        .set_index("timestamp")
        .sort_index()
    )
    return frame[["funding_rate", "premium_close"]]
