from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import time
import traceback

import pandas as pd

from paper_trading.approval import load_approved_builders
from paper_trading.config import PaperConfig, default_config
from paper_trading.engine import process_bar_roll
from paper_trading.frame import build_market_frame, ensure_warmup
from paper_trading.market_data import HyperliquidMarketData
from paper_trading.reporting import (
    append_decision,
    append_equity_point,
    append_fill,
    append_risk_event,
    write_daily_report,
    write_latest_status,
)
from paper_trading.risk import (
    closed_bar_is_recent,
    ensure_single_execution,
    halt_requested,
    quote_is_fresh,
)
from paper_trading.signals import latest_target_position
from paper_trading.state import load_or_initialize, save_state
from paper_trading.types import PaperRiskEvent, PaperRunSummary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one Hyperliquid mainnet paper-trading tick."
    )
    parser.add_argument("--artifacts-dir", type=Path)
    parser.add_argument("--deployment-root", type=Path)
    parser.add_argument("--base-url")
    parser.add_argument(
        "--loop", action="store_true", help="Poll continuously instead of running once."
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=30.0,
        help="Sleep duration between paper polling iterations when --loop is enabled.",
    )
    return parser.parse_args()


def state_path(artifacts_dir: Path) -> Path:
    return artifacts_dir / "runtime_state.json"


def closed_and_open_rows(
    recent_candles: pd.DataFrame,
) -> tuple[pd.Series, pd.Series | None]:
    closed = recent_candles[recent_candles["is_closed"]]
    if closed.empty:
        raise ValueError("no closed candles available for paper trading")
    latest_closed = closed.iloc[-1]
    execution_bar_time = latest_closed.name + pd.Timedelta(hours=1)
    current_row = recent_candles.loc[recent_candles.index == execution_bar_time]
    if current_row.empty:
        return latest_closed, None
    return latest_closed, current_row.iloc[-1]


def build_halted_summary(version_id: str, message: str, state) -> PaperRunSummary:
    return PaperRunSummary(
        version_id=version_id,
        status="halted",
        message=message,
        bar_close_time=None,
        current_qty=state.current_qty,
        gross_pnl_usdc=state.gross_pnl_usdc,
        funding_pnl_usdc=state.funding_pnl_usdc,
        fee_pnl_usdc=state.fee_pnl_usdc,
        slippage_pnl_usdc=state.slippage_pnl_usdc,
        net_pnl_usdc=state.net_pnl_usdc,
    )


def build_state_summary(
    version_id: str, status: str, message: str, bar_close_time: str | None, state
) -> PaperRunSummary:
    return PaperRunSummary(
        version_id=version_id,
        status=status,
        message=message,
        bar_close_time=bar_close_time,
        current_qty=state.current_qty,
        gross_pnl_usdc=state.gross_pnl_usdc,
        funding_pnl_usdc=state.funding_pnl_usdc,
        fee_pnl_usdc=state.fee_pnl_usdc,
        slippage_pnl_usdc=state.slippage_pnl_usdc,
        net_pnl_usdc=state.net_pnl_usdc,
    )


def write_risk_status(
    *,
    artifacts_dir: Path,
    session_date: str,
    summary: PaperRunSummary,
    state,
    event: PaperRiskEvent,
) -> PaperRunSummary:
    append_risk_event(artifacts_dir, event)
    write_latest_status(artifacts_dir, summary)
    write_daily_report(
        artifacts_dir,
        session_date=session_date,
        summary=summary,
        state=state,
    )
    return summary


def run_once(
    config: PaperConfig,
    *,
    artifacts_dir: Path | None = None,
    deployment_root: Path | None = None,
    market_data: HyperliquidMarketData | None = None,
    now: pd.Timestamp | None = None,
) -> PaperRunSummary:
    now = now or pd.Timestamp.now(tz="UTC")
    artifacts_dir = artifacts_dir or config.artifacts_dir
    deployment_root = deployment_root or config.deployment_root
    manifest, feature_builder, strategy_builder = load_approved_builders(
        deployment_root, config
    )
    state = load_or_initialize(state_path(artifacts_dir), manifest.version_id)
    halt_file = artifacts_dir / "HALT"
    if halt_requested(config, artifacts_dir=artifacts_dir):
        summary = build_halted_summary(
            manifest.version_id, f"halt file present at {halt_file}", state
        )
        return write_risk_status(
            artifacts_dir=artifacts_dir,
            session_date=now.date().isoformat(),
            summary=summary,
            state=state,
            event=PaperRiskEvent(
                version_id=manifest.version_id,
                timestamp=now.isoformat(),
                event="halt_requested",
                severity="warning",
                message=summary.message,
            ),
        )

    market_data = market_data or HyperliquidMarketData(
        symbol=config.symbol,
        interval=config.bar_interval,
        base_url=config.base_url,
    )
    closed_candles = market_data.bootstrap_candles(
        config.warmup_bars,
        max_backfill_bars=config.max_backfill_bars,
        now=now,
    )
    funding = market_data.fetch_funding_history(
        closed_candles.index.min(), end_time=now
    )
    market_frame = build_market_frame(closed_candles, funding=funding)
    ensure_warmup(market_frame, config.warmup_bars)
    target_position = latest_target_position(
        market_frame,
        feature_builder=feature_builder,
        strategy_builder=strategy_builder,
        symbol=config.symbol,
        interval=config.bar_interval,
    )

    recent_candles = market_data.poll_recent_candles(lookback_bars=3, now=now)
    latest_closed, current_row = closed_and_open_rows(recent_candles)
    if not closed_bar_is_recent(
        latest_closed.name + pd.Timedelta(hours=1), now, config
    ):
        summary = build_state_summary(
            manifest.version_id,
            "skip",
            "latest closed paper bar is stale",
            latest_closed.name.isoformat(),
            state,
        )
        return write_risk_status(
            artifacts_dir=artifacts_dir,
            session_date=now.date().isoformat(),
            summary=summary,
            state=state,
            event=PaperRiskEvent(
                version_id=manifest.version_id,
                timestamp=now.isoformat(),
                event="stale_closed_bar",
                severity="warning",
                message=summary.message,
                bar_close_time=latest_closed.name.isoformat(),
            ),
        )

    execution_bar_time = latest_closed.name + pd.Timedelta(hours=1)
    bbo = market_data.fetch_bbo(now=now)
    if not quote_is_fresh(bbo, now, config):
        bbo = None
    open_price = (
        float(current_row["open"])
        if current_row is not None
        else (
            (bbo.bid_px + bbo.ask_px) / 2.0
            if bbo is not None
            else float(latest_closed["close"])
        )
    )

    try:
        ensure_single_execution(state, execution_bar_time)
    except ValueError:
        summary = build_state_summary(
            manifest.version_id,
            "noop",
            "latest paper bar already processed",
            latest_closed.name.isoformat(),
            state,
        )
        write_latest_status(artifacts_dir, summary)
        return summary

    state, decision, fill = process_bar_roll(
        state=state,
        version_id=manifest.version_id,
        bar_close_time=latest_closed.name.isoformat(),
        execution_bar_time=execution_bar_time.isoformat(),
        target_position=target_position,
        current_open_price=open_price,
        bar_funding_rate=float(market_frame.iloc[-1]["funding_rate"]),
        bbo=bbo,
        config=config,
    )
    save_state(state_path(artifacts_dir), state)
    append_decision(artifacts_dir, decision)
    if fill is not None:
        append_fill(artifacts_dir, fill)
    if decision.status == "skip":
        append_risk_event(
            artifacts_dir,
            PaperRiskEvent(
                version_id=manifest.version_id,
                timestamp=decision.decision_time,
                event="paper_decision_skip",
                severity="warning",
                message=decision.reason,
                bar_close_time=decision.bar_close_time,
            ),
        )
    append_equity_point(artifacts_dir, now.isoformat(), state)
    summary = build_state_summary(
        manifest.version_id,
        "ok" if fill is not None else decision.status,
        decision.reason,
        latest_closed.name.isoformat(),
        state,
    )
    write_latest_status(artifacts_dir, summary)
    write_daily_report(
        artifacts_dir,
        session_date=now.date().isoformat(),
        summary=summary,
        state=state,
    )
    return summary


def render_summary(summary: PaperRunSummary) -> str:
    lines = [
        "---",
        f"version_id:     {summary.version_id}",
        f"status:         {summary.status}",
        f"message:        {summary.message}",
        f"bar_close_time: {summary.bar_close_time}",
        f"current_qty:    {summary.current_qty:.8f}",
        f"net_pnl_usdc:   {summary.net_pnl_usdc:.6f}",
    ]
    return "\n".join(lines)


def run_forever(
    config: PaperConfig,
    *,
    artifacts_dir: Path | None = None,
    deployment_root: Path | None = None,
    market_data: HyperliquidMarketData | None = None,
    poll_interval_seconds: float = 30.0,
    max_iterations: int | None = None,
    now_fn=lambda: pd.Timestamp.now(tz="UTC"),
    sleep_fn=time.sleep,
) -> PaperRunSummary:
    iteration = 0
    last_summary: PaperRunSummary | None = None
    last_exception: Exception | None = None
    while True:
        try:
            last_summary = run_once(
                config,
                artifacts_dir=artifacts_dir,
                deployment_root=deployment_root,
                market_data=market_data,
                now=now_fn(),
            )
            last_exception = None
            print(render_summary(last_summary), flush=True)
        except Exception as exc:
            last_exception = exc
            traceback.print_exc()
        iteration += 1
        if max_iterations is not None and iteration >= max_iterations:
            if last_summary is None and last_exception is not None:
                raise RuntimeError(
                    "paper loop exhausted max_iterations without a successful run"
                ) from last_exception
            return last_summary
        sleep_fn(poll_interval_seconds)


def main() -> int:
    args = parse_args()
    config = default_config()
    if args.base_url is not None:
        config = replace(config, base_url=args.base_url)
    if args.loop:
        summary = run_forever(
            config,
            artifacts_dir=args.artifacts_dir,
            deployment_root=args.deployment_root,
            poll_interval_seconds=args.poll_interval_seconds,
        )
    else:
        summary = run_once(
            config,
            artifacts_dir=args.artifacts_dir,
            deployment_root=args.deployment_root,
        )
    print(render_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
