from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from datetime import datetime, timezone
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
import webbrowser

import pandas as pd

from paper_trading.approval import load_approved_modules
from paper_trading.config import default_config
from paper_trading.frame import build_market_frame, ensure_warmup
from paper_trading.market_data import HyperliquidMarketData
from paper_trading.signals import make_market_bundle


FLOAT_FIELDS = {
    "target_position",
    "current_qty",
    "target_qty",
    "delta_qty",
    "open_price",
    "fill_price",
    "qty",
    "price",
    "notional_usdc",
    "fee_usdc",
    "slippage_usdc",
    "gross_pnl_usdc",
    "funding_pnl_usdc",
    "fee_pnl_usdc",
    "slippage_pnl_usdc",
    "net_pnl_usdc",
    "capital_usdc",
    "taker_fee_bps",
    "extra_slippage_bps",
    "last_mark_price",
}

HTML_TEMPLATE_PATH = Path(__file__).parent / "static" / "paper_dashboard.html"

SIGNAL_PATH_LABELS = {
    "slow_trend_full": "Slow trend full-size long",
    "slow_trend_stretched_partial": "Slow trend partial long",
    "breakout_reentry": "Breakout re-entry long",
    "slow_trend_partial": "Premium-limited partial long",
    "hard_blocked": "Risk override cancelled the long",
    "volatility_blocked": "Volatility gate blocked entry",
    "volatility_ready_no_entry": "Volatility passed but no entry path fired",
    "flat_no_entry": "No entry path fired",
    "test_long": "Test long signal",
}

STRATEGY_DETAIL_FIELDS = {
    "trend_regime_7d",
    "trend_regime",
    "trend_slope",
    "volatility_14d",
    "volatility_30d",
    "funding_latest",
    "premium_7d",
    "breakout_20d",
}


def _bool_flag(value: Any) -> bool:
    return bool(value)


def _format_pct(value: Any, digits: int = 2, empty: str = "-") -> str:
    if value is None:
        return empty
    try:
        number = float(value)
    except (TypeError, ValueError):
        return empty
    return f"{number * 100:.{digits}f}%"


def _signal_path_label(value: Any) -> str:
    label = str(value or "").strip()
    if not label:
        return "Signal path unavailable"
    return SIGNAL_PATH_LABELS.get(label, label.replace("_", " "))


def _condition(label: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"label": label, "passed": passed, "detail": detail}


def _generic_signal_story(signal_row: dict[str, Any]) -> dict[str, Any]:
    transition = str(signal_row.get("transition") or "unknown")
    state_before = float(signal_row.get("state_before") or 0.0)
    effective_position = float(signal_row.get("effective_position") or 0.0)
    cooldown_remaining = int(signal_row.get("cooldown_hours_remaining") or 0)
    path_label = _signal_path_label(signal_row.get("signal_path"))

    if transition == "enter_long":
        headline = "Long entry fired"
        summary = f"Traded because {path_label.lower()} passed with cooldown clear."
    elif transition == "exit_long":
        headline = "Exit to flat fired"
        summary = "Traded out because the state machine dropped its target to flat."
    elif transition == "entry_cooldown":
        headline = "Entry signal is waiting on cooldown"
        summary = f"A long setup is live, but {cooldown_remaining}h of cooldown remain."
    elif transition == "exit_cooldown":
        headline = "Exit signal is waiting on cooldown"
        summary = f"The strategy wants to flatten, but {cooldown_remaining}h of cooldown remain."
    elif transition == "hold_long":
        headline = "Holding current long"
        summary = "No new trade because the state machine is already long."
    else:
        headline = "No trade"
        summary = "No entry path was active on the latest processed bar."

    blockers: list[dict[str, Any]] = []
    if transition in {"entry_cooldown", "exit_cooldown"}:
        blockers.append(
            _condition(
                "Cooldown gate",
                False,
                f"{cooldown_remaining}h remain before another state change is allowed.",
            )
        )
    elif transition == "hold_long" and state_before != 0.0:
        blockers.append(
            _condition(
                "Already in position",
                False,
                "This state machine only exits on a zero target and does not resize between non-zero targets.",
            )
        )

    return {
        "headline": headline,
        "summary": summary,
        "path_label": path_label,
        "transition": transition,
        "state_before": state_before,
        "effective_position": effective_position,
        "candidate_target": float(signal_row.get("candidate_target") or 0.0),
        "cooldown_hours_remaining": cooldown_remaining,
        "supporting_conditions": [],
        "blocking_conditions": blockers,
        "metrics": [],
    }


def _detailed_signal_story(signal_row: dict[str, Any]) -> dict[str, Any]:
    transition = str(signal_row.get("transition") or "unknown")
    path_label = _signal_path_label(signal_row.get("signal_path"))
    signal_target = float(signal_row.get("signal_target") or 0.0)
    candidate_target = float(signal_row.get("candidate_target") or 0.0)
    cooldown_ready = _bool_flag(signal_row.get("cooldown_ready"))
    cooldown_remaining = int(signal_row.get("cooldown_hours_remaining") or 0)
    hard_block = _bool_flag(signal_row.get("hard_block"))
    state_before = float(signal_row.get("state_before") or 0.0)
    effective_position = float(signal_row.get("effective_position") or 0.0)
    trade_intent = _bool_flag(signal_row.get("trade_intent"))

    passes = [
        _condition(
            "Volatility gate",
            _bool_flag(signal_row.get("volatility_ready")),
            (
                f"14d {_format_pct(signal_row.get('volatility_14d'))}, "
                f"30d {_format_pct(signal_row.get('volatility_30d'))}, cap 0.60%."
            ),
        ),
        _condition(
            "Slow trend entry",
            _bool_flag(signal_row.get("slow_trend_ready")),
            (
                f"Trend {_format_pct(signal_row.get('trend_regime'))} vs >= 0.90%, "
                f"slope {_format_pct(signal_row.get('trend_slope'))} vs > -0.50%, "
                f"premium {_format_pct(signal_row.get('premium_7d'), 3)} vs >= -0.015%."
            ),
        ),
        _condition(
            "Breakout re-entry",
            _bool_flag(signal_row.get("breakout_reentry")),
            (
                f"20d breakout {_format_pct(signal_row.get('breakout_20d'))} vs > -1.20%, "
                f"7d regime {_format_pct(signal_row.get('trend_regime_7d'))} vs > 0.70%."
            ),
        ),
        _condition(
            "Partial fallback",
            _bool_flag(signal_row.get("partial_ready")),
            (
                f"Premium {_format_pct(signal_row.get('premium_7d'), 3)} vs >= -0.030% "
                "with trend precheck still passing."
            ),
        ),
        _condition(
            "Risk override clear",
            not hard_block,
            (
                f"Funding {_format_pct(signal_row.get('funding_latest'), 3)} vs < 0.090%, "
                f"slope {_format_pct(signal_row.get('trend_slope'))} vs >= -1.00%."
            ),
        ),
    ]

    if trade_intent or transition in {"entry_cooldown", "exit_cooldown"}:
        passes.append(
            _condition(
                "Cooldown gate",
                cooldown_ready,
                (
                    "73h state-change cooldown is clear."
                    if cooldown_ready
                    else f"{cooldown_remaining}h remain before another state change."
                ),
            )
        )

    supporting_conditions = [item for item in passes if item["passed"]]
    blocking_conditions = [item for item in passes if not item["passed"]]

    if transition == "enter_long":
        headline = "Long entry fired"
        summary = f"Traded because {path_label.lower()} passed and no guardrail cancelled the setup."
    elif transition == "exit_long":
        headline = "Exit to flat fired"
        summary = "Traded out because the strategy target fell to flat and cooldown was already satisfied."
    elif transition == "entry_cooldown":
        headline = "Long setup is waiting on cooldown"
        summary = f"{path_label} is active, but {cooldown_remaining}h of cooldown remain before entry."
    elif transition == "exit_cooldown":
        headline = "Exit setup is waiting on cooldown"
        summary = f"The strategy wants to flatten, but {cooldown_remaining}h of cooldown remain."
    elif transition == "hold_long" and candidate_target != 0.0:
        headline = "Holding current long"
        summary = "No new trade because the strategy is already long and only exits on a zero target."
        blocking_conditions.insert(
            0,
            _condition(
                "Already in position",
                False,
                "Non-zero to non-zero target changes do not trigger a resize in this state machine.",
            ),
        )
    elif hard_block and signal_target > 0.0:
        headline = "Risk override cancelled the setup"
        summary = "A long path lit up, but the hard risk block zeroed the target before execution."
    else:
        headline = "No trade"
        summary = "No entry path reached its threshold on the latest processed bar."

    metrics = [
        {"label": "Trend regime", "value": _format_pct(signal_row.get("trend_regime"))},
        {"label": "Trend slope", "value": _format_pct(signal_row.get("trend_slope"))},
        {"label": "7d regime", "value": _format_pct(signal_row.get("trend_regime_7d"))},
        {"label": "20d breakout", "value": _format_pct(signal_row.get("breakout_20d"))},
        {"label": "14d vol", "value": _format_pct(signal_row.get("volatility_14d"))},
        {"label": "30d vol", "value": _format_pct(signal_row.get("volatility_30d"))},
        {"label": "Funding", "value": _format_pct(signal_row.get("funding_latest"), 3)},
        {"label": "7d premium", "value": _format_pct(signal_row.get("premium_7d"), 3)},
    ]

    return {
        "headline": headline,
        "summary": summary,
        "path_label": path_label,
        "transition": transition,
        "state_before": state_before,
        "effective_position": effective_position,
        "candidate_target": candidate_target,
        "cooldown_hours_remaining": cooldown_remaining,
        "supporting_conditions": supporting_conditions,
        "blocking_conditions": blocking_conditions,
        "metrics": metrics,
    }


def _build_signal_story(signal_row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not signal_row:
        return None
    if STRATEGY_DETAIL_FIELDS.issubset(signal_row):
        return _detailed_signal_story(signal_row)
    return _generic_signal_story(signal_row)


def parse_args() -> argparse.Namespace:
    config = default_config()
    parser = argparse.ArgumentParser(
        description="Run a local dashboard for Hyperliquid paper-trading artifacts."
    )
    parser.add_argument("--artifacts-dir", type=Path, default=config.artifacts_dir)
    parser.add_argument("--deployment-root", type=Path, default=config.deployment_root)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--refresh-seconds",
        type=float,
        default=15.0,
        help="Browser polling interval for dashboard refreshes.",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the dashboard URL in the default browser after startup.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _coerce_value(field: str, value: str | None) -> Any:
    if value is None or value == "":
        return None
    if field in FLOAT_FIELDS:
        return float(value)
    return value


def _read_tsv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [
            {field: _coerce_value(field, value) for field, value in row.items()}
            for row in reader
        ]


def _read_daily_reports(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    reports: list[dict[str, Any]] = []
    for report_path in sorted(path.glob("*.json")):
        payload = _read_json(report_path) or {}
        reports.append(
            {
                "session_date": report_path.stem,
                "summary": payload.get("summary"),
                "state": payload.get("state"),
            }
        )
    return sorted(reports, key=lambda row: row["session_date"], reverse=True)


def _timestamp_from_mtime(path: Path) -> str | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _file_meta(path: Path, *, row_count: int | None = None) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "modified_at": _timestamp_from_mtime(path),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "row_count": row_count,
    }


def _latest_timestamp(*values: str | None) -> str | None:
    present = [value for value in values if value]
    if not present:
        return None
    return max(present)


def _max_drawdown(values: list[float]) -> float:
    peak = 0.0
    max_drawdown = 0.0
    for value in values:
        peak = max(peak, value)
        max_drawdown = min(max_drawdown, value - peak)
    return max_drawdown


def _position_side(current_qty: float) -> str:
    if current_qty > 0:
        return "long"
    if current_qty < 0:
        return "short"
    return "flat"


@lru_cache(maxsize=8)
def _compute_live_signal_rows(
    deployment_root: str,
    artifacts_dir: str,
    version_id: str,
    latest_bar_close_time: str | None,
) -> list[dict[str, Any]]:
    del version_id
    del latest_bar_close_time
    config = replace(
        default_config(),
        deployment_root=Path(deployment_root),
        artifacts_dir=Path(artifacts_dir),
    )
    manifest, features_module, strategy_module = load_approved_modules(
        config.deployment_root, config
    )
    feature_builder = getattr(features_module, "build_features", None)
    strategy_builder = getattr(strategy_module, "generate_position", None)
    signal_describer = getattr(strategy_module, "describe_signal", None)
    if feature_builder is None or strategy_builder is None:
        raise AttributeError("approved snapshot is missing build_features or generate_position")
    if signal_describer is None:
        raise AttributeError("approved snapshot is missing describe_signal")

    now = pd.Timestamp.now(tz="UTC")
    market_data = HyperliquidMarketData(
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
    market = make_market_bundle(
        market_frame, symbol=manifest.symbol, interval=manifest.bar_interval
    )
    features = feature_builder(market)
    positions = strategy_builder(features, market)
    diagnostics = signal_describer(features, market).copy()
    diagnostics["position"] = positions
    diagnostics["bar_close_time"] = [
        str(value)
        for value in diagnostics.get(
            "bar_close_time",
            pd.Series(diagnostics.index.map(lambda value: value.isoformat()), index=diagnostics.index),
        )
    ]
    return diagnostics.reset_index(drop=True).to_dict(orient="records")


def _load_signal_payload(
    *,
    artifacts_dir: Path,
    deployment_root: Path,
    current_deployment: dict[str, Any] | None,
    latest_status: dict[str, Any] | None,
    decisions: list[dict[str, Any]],
    max_rows: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    version_id = str((current_deployment or {}).get("version_id") or "").strip()
    latest_bar_close_time = (
        (latest_status or {}).get("bar_close_time")
        or (decisions[0].get("bar_close_time") if decisions else None)
    )
    if not version_id:
        return {
            "available": False,
            "message": "No approved deployment is active, so signal diagnostics are unavailable.",
        }, {}

    try:
        signal_rows = _compute_live_signal_rows(
            str(deployment_root),
            str(artifacts_dir),
            version_id,
            str(latest_bar_close_time or ""),
        )
    except Exception as exc:
        return {
            "available": False,
            "message": f"Signal diagnostics unavailable: {exc}",
        }, {}

    by_bar_close_time = {
        str(row.get("bar_close_time")): row for row in signal_rows if row.get("bar_close_time")
    }
    latest_row = (
        by_bar_close_time.get(str(latest_bar_close_time))
        or (signal_rows[-1] if signal_rows else None)
    )
    latest_story = _build_signal_story(latest_row)
    recent_map: dict[str, dict[str, Any]] = {}
    for decision in decisions[:max_rows]:
        signal_row = by_bar_close_time.get(str(decision.get("bar_close_time") or ""))
        story = _build_signal_story(signal_row)
        if story is None:
            continue
        recent_map[str(decision.get("bar_close_time") or decision.get("decision_time") or "")] = story

    return (
        {
            "available": latest_story is not None,
            "message": None if latest_story is not None else "No signal diagnostics were computed.",
            "latest": latest_story,
        },
        recent_map,
    )


def load_dashboard_payload(
    artifacts_dir: Path,
    deployment_root: Path,
    *,
    max_rows: int = 200,
) -> dict[str, Any]:
    latest_status_path = artifacts_dir / "latest_status.json"
    runtime_state_path = artifacts_dir / "runtime_state.json"
    decisions_path = artifacts_dir / "decisions.tsv"
    fills_path = artifacts_dir / "fills.tsv"
    equity_path = artifacts_dir / "equity.tsv"
    risk_events_path = artifacts_dir / "risk_events.tsv"
    daily_dir = artifacts_dir / "daily"

    latest_status = _read_json(latest_status_path)
    runtime_state = _read_json(runtime_state_path)
    current_deployment = _read_json(deployment_root / "current.json")
    manifest = None
    if current_deployment and current_deployment.get("version_id"):
        manifest = _read_json(
            deployment_root
            / "versions"
            / current_deployment["version_id"]
            / "manifest.json"
        )

    decisions = sorted(
        _read_tsv(decisions_path),
        key=lambda row: row.get("decision_time") or "",
        reverse=True,
    )
    fills = sorted(
        _read_tsv(fills_path),
        key=lambda row: row.get("timestamp") or "",
        reverse=True,
    )
    risk_events = sorted(
        _read_tsv(risk_events_path),
        key=lambda row: row.get("timestamp") or "",
        reverse=True,
    )
    equity_points = sorted(_read_tsv(equity_path), key=lambda row: row.get("timestamp") or "")
    daily_reports = _read_daily_reports(daily_dir)

    signal_payload, recent_signal_map = _load_signal_payload(
        artifacts_dir=artifacts_dir,
        deployment_root=deployment_root,
        current_deployment=current_deployment,
        latest_status=latest_status,
        decisions=decisions,
        max_rows=max_rows,
    )
    recent_decisions = []
    for row in decisions[:max_rows]:
        signal_key = str(row.get("bar_close_time") or row.get("decision_time") or "")
        enriched = dict(row)
        if signal_key in recent_signal_map:
            enriched["signal_story"] = recent_signal_map[signal_key]
        recent_decisions.append(enriched)

    state_source = runtime_state or latest_status or {}
    current_qty = float(state_source.get("current_qty") or 0.0)
    last_mark_price = float((runtime_state or {}).get("last_mark_price") or 0.0)
    current_notional = abs(current_qty) * last_mark_price if last_mark_price else 0.0
    net_series = [float(point.get("net_pnl_usdc") or 0.0) for point in equity_points]
    turnover_usdc = sum(float(fill.get("notional_usdc") or 0.0) for fill in fills)

    latest_activity_time = _latest_timestamp(
        (decisions[0].get("decision_time") if decisions else None),
        (fills[0].get("timestamp") if fills else None),
        (risk_events[0].get("timestamp") if risk_events else None),
        (latest_status or {}).get("bar_close_time"),
        _timestamp_from_mtime(latest_status_path),
        _timestamp_from_mtime(runtime_state_path),
    )

    return {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "paths": {
            "artifacts_dir": str(artifacts_dir),
            "deployment_root": str(deployment_root),
        },
        "health": {
            "halt_requested": (artifacts_dir / "HALT").exists(),
            "latest_activity_time": latest_activity_time,
            "runner_status": (latest_status or {}).get("status", "missing"),
            "message": (latest_status or {}).get(
                "message", "No paper status has been written yet."
            ),
        },
        "current_deployment": current_deployment,
        "manifest": manifest,
        "latest_status": latest_status,
        "runtime_state": runtime_state,
        "stats": {
            "current_position_side": _position_side(current_qty),
            "current_notional_usdc": current_notional,
            "decision_count": len(decisions),
            "trade_count": sum(1 for row in decisions if row.get("status") == "trade"),
            "fill_count": len(fills),
            "risk_event_count": len(risk_events),
            "daily_report_count": len(daily_reports),
            "turnover_usdc": turnover_usdc,
            "equity_high_watermark_usdc": max([0.0, *net_series]),
            "equity_low_watermark_usdc": min([0.0, *net_series]),
            "max_drawdown_usdc": _max_drawdown(net_series),
        },
        "files": {
            "latest_status": _file_meta(latest_status_path),
            "runtime_state": _file_meta(runtime_state_path),
            "decisions": _file_meta(decisions_path, row_count=len(decisions)),
            "fills": _file_meta(fills_path, row_count=len(fills)),
            "equity": _file_meta(equity_path, row_count=len(equity_points)),
            "risk_events": _file_meta(risk_events_path, row_count=len(risk_events)),
            "daily": _file_meta(daily_dir, row_count=len(daily_reports)),
        },
        "equity_points": equity_points,
        "signal_state": signal_payload,
        "recent_decisions": recent_decisions,
        "recent_fills": fills[:max_rows],
        "recent_risk_events": risk_events[:max_rows],
        "daily_reports": daily_reports[:max_rows],
    }


@lru_cache(maxsize=1)
def load_html_template() -> str:
    return HTML_TEMPLATE_PATH.read_text(encoding="utf-8")


class DashboardApp:
    def __init__(
        self,
        *,
        artifacts_dir: Path,
        deployment_root: Path,
        refresh_interval_ms: int = 15_000,
    ) -> None:
        self.artifacts_dir = artifacts_dir
        self.deployment_root = deployment_root
        self.refresh_interval_ms = refresh_interval_ms

    def render_index(self) -> bytes:
        template = load_html_template()
        html = template.replace("__REFRESH_INTERVAL_MS__", str(self.refresh_interval_ms))
        return html.encode("utf-8")

    def route(self, path: str) -> tuple[HTTPStatus, str, bytes]:
        if path in {"/", "/index.html"}:
            return (
                HTTPStatus.OK,
                "text/html; charset=utf-8",
                self.render_index(),
            )
        if path == "/api/dashboard":
            payload = load_dashboard_payload(self.artifacts_dir, self.deployment_root)
            return (
                HTTPStatus.OK,
                "application/json; charset=utf-8",
                json.dumps(payload, indent=2).encode("utf-8"),
            )
        if path == "/healthz":
            return HTTPStatus.OK, "text/plain; charset=utf-8", b"ok"
        return HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", b"Not found"


class DashboardRequestHandler(BaseHTTPRequestHandler):
    app: DashboardApp

    def _respond(self, *, include_body: bool) -> None:
        status, content_type, body = self.app.route(urlparse(self.path).path)
        self.send_response(int(status))
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if include_body:
            self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        self._respond(include_body=True)

    def do_HEAD(self) -> None:  # noqa: N802
        self._respond(include_body=False)

    def log_message(self, format: str, *args: object) -> None:
        return


def build_server(
    app: DashboardApp,
    *,
    host: str,
    port: int,
) -> ThreadingHTTPServer:
    handler = type("PaperDashboardHandler", (DashboardRequestHandler,), {"app": app})
    return ThreadingHTTPServer((host, port), handler)


def main() -> None:
    args = parse_args()
    refresh_interval_ms = max(int(args.refresh_seconds * 1000), 1_000)
    app = DashboardApp(
        artifacts_dir=args.artifacts_dir,
        deployment_root=args.deployment_root,
        refresh_interval_ms=refresh_interval_ms,
    )
    server = build_server(app, host=args.host, port=args.port)
    url = f"http://{args.host}:{args.port}"
    print(f"paper dashboard listening on {url}")
    if args.open_browser:
        webbrowser.open(url)
    with server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("paper dashboard stopped")


if __name__ == "__main__":
    main()
