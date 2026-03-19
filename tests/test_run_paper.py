from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from paper_trading.config import PaperConfig
from paper_trading.runner import render_summary, run_forever, run_once
from paper_trading.types import BboSnapshot, PaperRunSummary
from tests.paper_helpers import make_candles, write_approved_snapshot


class StubMarketData:
    def __init__(self, closed_candles: pd.DataFrame, recent_candles: pd.DataFrame):
        self._closed_candles = closed_candles
        self._recent_candles = recent_candles

    def bootstrap_candles(
        self,
        warmup_bars: int,
        *,
        max_backfill_bars: int = 5_000,
        now: pd.Timestamp | None = None,
        include_incomplete: bool = False,
    ) -> pd.DataFrame:
        limit = min(warmup_bars, max_backfill_bars)
        return self._closed_candles.iloc[-limit:]

    def fetch_funding_history(
        self, start_time: pd.Timestamp, *, end_time: pd.Timestamp | None = None
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {"funding_rate": [0.0, 0.0, 0.0], "premium_close": [0.0, 0.0, 0.0]},
            index=self._closed_candles.index[-3:],
        )

    def poll_recent_candles(
        self, lookback_bars: int = 3, *, now: pd.Timestamp | None = None
    ) -> pd.DataFrame:
        return self._recent_candles.iloc[-lookback_bars:]

    def fetch_bbo(self, *, now: pd.Timestamp | None = None) -> BboSnapshot | None:
        timestamp = (now or pd.Timestamp.now(tz="UTC")).isoformat()
        return BboSnapshot(timestamp=timestamp, bid_px=102.9, ask_px=103.1)


class DelayedBboMarketData(StubMarketData):
    def __init__(self, closed_candles: pd.DataFrame, recent_candles: pd.DataFrame):
        super().__init__(closed_candles, recent_candles)
        self._bbo_calls = 0

    def fetch_bbo(self, *, now: pd.Timestamp | None = None) -> BboSnapshot | None:
        self._bbo_calls += 1
        if self._bbo_calls == 1:
            return None
        return super().fetch_bbo(now=now)


def test_run_once_writes_paper_artifacts_and_is_idempotent(tmp_path: Path) -> None:
    deployment_root = tmp_path / "deployments" / "paper"
    artifacts_dir = tmp_path / "paper_artifacts"
    config = PaperConfig(
        deployment_root=deployment_root,
        artifacts_dir=artifacts_dir,
        warmup_bars=3,
        stale_quote_seconds=60,
    )
    write_approved_snapshot(deployment_root, config)
    closed_index = pd.date_range("2026-03-18 00:00:00", periods=3, freq="1h", tz="UTC")
    open_index = pd.DatetimeIndex([closed_index[-1] + pd.Timedelta(hours=1)], tz="UTC")
    closed_candles = make_candles(closed_index, closed=True)
    recent_candles = pd.concat([closed_candles, make_candles(open_index, closed=False)])
    market_data = StubMarketData(closed_candles, recent_candles)
    now = pd.Timestamp("2026-03-18T03:00:10+00:00")

    summary = run_once(
        config,
        artifacts_dir=artifacts_dir,
        deployment_root=deployment_root,
        market_data=market_data,
        now=now,
    )
    second = run_once(
        config,
        artifacts_dir=artifacts_dir,
        deployment_root=deployment_root,
        market_data=market_data,
        now=now,
    )

    assert summary.status == "ok"
    assert (artifacts_dir / "decisions.tsv").exists()
    assert (artifacts_dir / "fills.tsv").exists()
    assert (artifacts_dir / "runtime_state.json").exists()
    assert (artifacts_dir / "daily" / "2026-03-18.json").exists()
    assert second.status == "noop"


def test_run_forever_repeats_until_max_iterations(tmp_path: Path) -> None:
    deployment_root = tmp_path / "deployments" / "paper"
    artifacts_dir = tmp_path / "paper_artifacts"
    config = PaperConfig(
        deployment_root=deployment_root,
        artifacts_dir=artifacts_dir,
        warmup_bars=3,
        stale_quote_seconds=60,
    )
    write_approved_snapshot(deployment_root, config)
    closed_index = pd.date_range("2026-03-18 00:00:00", periods=3, freq="1h", tz="UTC")
    open_index = pd.DatetimeIndex([closed_index[-1] + pd.Timedelta(hours=1)], tz="UTC")
    closed_candles = make_candles(closed_index, closed=True)
    recent_candles = pd.concat([closed_candles, make_candles(open_index, closed=False)])
    market_data = StubMarketData(closed_candles, recent_candles)
    sleeps: list[float] = []

    summary = run_forever(
        config,
        artifacts_dir=artifacts_dir,
        deployment_root=deployment_root,
        market_data=market_data,
        poll_interval_seconds=12.5,
        max_iterations=2,
        now_fn=lambda: pd.Timestamp("2026-03-18T03:00:10+00:00"),
        sleep_fn=sleeps.append,
    )

    assert summary.status == "noop"
    assert sleeps == [12.5]


def test_run_once_honors_halt_file(tmp_path: Path) -> None:
    deployment_root = tmp_path / "deployments" / "paper"
    artifacts_dir = tmp_path / "paper_artifacts"
    config = PaperConfig(
        deployment_root=deployment_root,
        artifacts_dir=artifacts_dir,
        warmup_bars=3,
        stale_quote_seconds=60,
    )
    write_approved_snapshot(deployment_root, config)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "HALT").write_text("stop", encoding="utf-8")
    closed_index = pd.date_range("2026-03-18 00:00:00", periods=3, freq="1h", tz="UTC")
    open_index = pd.DatetimeIndex([closed_index[-1] + pd.Timedelta(hours=1)], tz="UTC")
    closed_candles = make_candles(closed_index, closed=True)
    recent_candles = pd.concat([closed_candles, make_candles(open_index, closed=False)])
    market_data = StubMarketData(closed_candles, recent_candles)

    summary = run_once(
        config,
        artifacts_dir=artifacts_dir,
        deployment_root=deployment_root,
        market_data=market_data,
        now=pd.Timestamp("2026-03-18T03:00:10+00:00"),
    )

    assert summary.status == "halted"
    assert (artifacts_dir / "risk_events.tsv").exists()
    assert (artifacts_dir / "daily" / "2026-03-18.json").exists()


def test_run_once_retries_same_bar_after_skip_when_bbo_recovers(tmp_path: Path) -> None:
    deployment_root = tmp_path / "deployments" / "paper"
    artifacts_dir = tmp_path / "paper_artifacts"
    config = PaperConfig(
        deployment_root=deployment_root,
        artifacts_dir=artifacts_dir,
        warmup_bars=3,
        stale_quote_seconds=60,
    )
    write_approved_snapshot(deployment_root, config)
    closed_index = pd.date_range("2026-03-18 00:00:00", periods=3, freq="1h", tz="UTC")
    open_index = pd.DatetimeIndex([closed_index[-1] + pd.Timedelta(hours=1)], tz="UTC")
    closed_candles = make_candles(closed_index, closed=True)
    recent_candles = pd.concat([closed_candles, make_candles(open_index, closed=False)])
    market_data = DelayedBboMarketData(closed_candles, recent_candles)
    now = pd.Timestamp("2026-03-18T03:00:10+00:00")

    first = run_once(
        config,
        artifacts_dir=artifacts_dir,
        deployment_root=deployment_root,
        market_data=market_data,
        now=now,
    )
    second = run_once(
        config,
        artifacts_dir=artifacts_dir,
        deployment_root=deployment_root,
        market_data=market_data,
        now=now,
    )

    assert first.status == "skip"
    assert second.status == "ok"
    decisions = pd.read_csv(artifacts_dir / "decisions.tsv", sep="\t")
    assert list(decisions["status"]) == ["skip", "trade"]


def test_run_forever_survives_transient_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}
    sleeps: list[float] = []
    summary = PaperRunSummary(
        version_id="v1",
        status="ok",
        message="ok",
        bar_close_time="2026-03-18T00:00:00+00:00",
        current_qty=1.0,
        gross_pnl_usdc=0.0,
        funding_pnl_usdc=0.0,
        fee_pnl_usdc=0.0,
        slippage_pnl_usdc=0.0,
        net_pnl_usdc=0.0,
    )

    def flaky_run_once(*args, **kwargs) -> PaperRunSummary:
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("temporary network issue")
        return summary

    monkeypatch.setattr("paper_trading.runner.run_once", flaky_run_once)

    result = run_forever(
        PaperConfig(),
        poll_interval_seconds=12.5,
        max_iterations=2,
        sleep_fn=sleeps.append,
    )

    assert result == summary
    assert sleeps == [12.5]


def test_render_summary_includes_status_and_pnl() -> None:
    rendered = render_summary(
        PaperRunSummary(
            version_id="v1",
            status="ok",
            message="ok",
            bar_close_time="2026-03-18T00:00:00+00:00",
            current_qty=1.25,
            gross_pnl_usdc=10.0,
            funding_pnl_usdc=1.0,
            fee_pnl_usdc=-0.5,
            slippage_pnl_usdc=-0.25,
            net_pnl_usdc=12.5,
        )
    )

    assert "status:         ok" in rendered
    assert "net_pnl_usdc:   12.500000" in rendered
