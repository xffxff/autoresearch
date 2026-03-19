from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from paper_trading.types import (
    PaperDecision,
    PaperFill,
    PaperRiskEvent,
    PaperRunSummary,
    PaperState,
)


DECISION_HEADER = [
    "version_id",
    "bar_close_time",
    "execution_bar_time",
    "decision_time",
    "target_position",
    "current_qty",
    "target_qty",
    "delta_qty",
    "open_price",
    "fill_price",
    "status",
    "action",
    "reason",
]

FILL_HEADER = [
    "version_id",
    "timestamp",
    "execution_bar_time",
    "side",
    "qty",
    "price",
    "notional_usdc",
    "fee_usdc",
    "slippage_usdc",
]

EQUITY_HEADER = [
    "version_id",
    "timestamp",
    "current_qty",
    "gross_pnl_usdc",
    "funding_pnl_usdc",
    "fee_pnl_usdc",
    "slippage_pnl_usdc",
    "net_pnl_usdc",
]

RISK_EVENT_HEADER = [
    "version_id",
    "timestamp",
    "event",
    "severity",
    "message",
    "bar_close_time",
]


def _append_row(path: Path, header: list[str], row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_decision(artifacts_dir: Path, decision: PaperDecision) -> None:
    _append_row(artifacts_dir / "decisions.tsv", DECISION_HEADER, asdict(decision))


def append_fill(artifacts_dir: Path, fill: PaperFill) -> None:
    _append_row(artifacts_dir / "fills.tsv", FILL_HEADER, asdict(fill))


def append_equity_point(artifacts_dir: Path, timestamp: str, state: PaperState) -> None:
    row = {
        "version_id": state.version_id,
        "timestamp": timestamp,
        "current_qty": state.current_qty,
        "gross_pnl_usdc": state.gross_pnl_usdc,
        "funding_pnl_usdc": state.funding_pnl_usdc,
        "fee_pnl_usdc": state.fee_pnl_usdc,
        "slippage_pnl_usdc": state.slippage_pnl_usdc,
        "net_pnl_usdc": state.net_pnl_usdc,
    }
    _append_row(artifacts_dir / "equity.tsv", EQUITY_HEADER, row)


def append_risk_event(artifacts_dir: Path, event: PaperRiskEvent) -> None:
    _append_row(artifacts_dir / "risk_events.tsv", RISK_EVENT_HEADER, asdict(event))


def write_latest_status(artifacts_dir: Path, summary: PaperRunSummary) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "latest_status.json").write_text(
        json.dumps(asdict(summary), indent=2), encoding="utf-8"
    )


def write_daily_report(
    artifacts_dir: Path,
    *,
    session_date: str,
    summary: PaperRunSummary,
    state: PaperState,
) -> None:
    report_dir = artifacts_dir / "daily"
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": asdict(summary),
        "state": asdict(state),
    }
    (report_dir / f"{session_date}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
