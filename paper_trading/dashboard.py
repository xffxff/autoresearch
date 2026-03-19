from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
import webbrowser

from paper_trading.config import default_config


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
        "recent_decisions": decisions[:max_rows],
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
