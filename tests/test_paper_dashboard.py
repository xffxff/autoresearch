from __future__ import annotations

import json
from http import HTTPStatus
from pathlib import Path

import pytest

from paper_trading.config import PaperConfig
from paper_trading.dashboard import DashboardApp, load_dashboard_payload
from paper_trading.reporting import (
    append_decision,
    append_equity_point,
    append_fill,
    append_risk_event,
    write_daily_report,
    write_latest_status,
)
from paper_trading.state import save_state
from paper_trading.types import (
    PaperDecision,
    PaperFill,
    PaperRiskEvent,
    PaperRunSummary,
    PaperState,
)
from tests.paper_helpers import write_approved_snapshot


def stub_signal_payload(*args, **kwargs) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    del args
    del kwargs
    story = {
        "headline": "Long entry fired",
        "summary": "Traded because the long path passed and no blocker cancelled the setup.",
        "path_label": "Test long signal",
        "transition": "enter_long",
        "state_before": 0.0,
        "effective_position": 1.0,
        "candidate_target": 1.0,
        "cooldown_hours_remaining": 0,
        "supporting_conditions": [
            {"label": "Volatility gate", "passed": True, "detail": "Passed."}
        ],
        "blocking_conditions": [],
        "metrics": [{"label": "Signal", "value": "1.00"}],
    }
    return {
        "available": True,
        "message": None,
        "latest": story,
    }, {"2026-03-19T13:00:00+00:00": story}


def write_dashboard_fixture(tmp_path: Path) -> tuple[Path, Path]:
    deployment_root = tmp_path / "deployments" / "paper"
    artifacts_dir = tmp_path / "paper_artifacts"
    config = PaperConfig(deployment_root=deployment_root, artifacts_dir=artifacts_dir)
    write_approved_snapshot(deployment_root, config, version_id="paper-test-v1")

    state = PaperState(
        version_id="paper-test-v1",
        current_qty=0.125,
        last_bar_time="2026-03-19T14:00:00+00:00",
        last_mark_price=72_000.0,
        gross_pnl_usdc=150.0,
        funding_pnl_usdc=-2.5,
        fee_pnl_usdc=-5.0,
        slippage_pnl_usdc=-7.5,
        net_pnl_usdc=135.0,
    )
    save_state(artifacts_dir / "runtime_state.json", state)

    summary = PaperRunSummary(
        version_id="paper-test-v1",
        status="ok",
        message="ok",
        bar_close_time="2026-03-19T13:00:00+00:00",
        current_qty=state.current_qty,
        gross_pnl_usdc=state.gross_pnl_usdc,
        funding_pnl_usdc=state.funding_pnl_usdc,
        fee_pnl_usdc=state.fee_pnl_usdc,
        slippage_pnl_usdc=state.slippage_pnl_usdc,
        net_pnl_usdc=state.net_pnl_usdc,
    )
    write_latest_status(artifacts_dir, summary)
    append_decision(
        artifacts_dir,
        PaperDecision(
            version_id="paper-test-v1",
            bar_close_time="2026-03-19T13:00:00+00:00",
            execution_bar_time="2026-03-19T14:00:00+00:00",
            decision_time="2026-03-19T14:05:00+00:00",
            target_position=1.0,
            current_qty=0.0,
            target_qty=0.125,
            delta_qty=0.125,
            open_price=71_500.0,
            fill_price=72_000.0,
            status="trade",
            action="buy",
            reason="ok",
        ),
    )
    append_fill(
        artifacts_dir,
        PaperFill(
            version_id="paper-test-v1",
            timestamp="2026-03-19T14:05:00+00:00",
            execution_bar_time="2026-03-19T14:00:00+00:00",
            side="buy",
            qty=0.125,
            price=72_000.0,
            notional_usdc=9_000.0,
            fee_usdc=4.5,
            slippage_usdc=7.5,
        ),
    )
    append_equity_point(artifacts_dir, "2026-03-19T14:05:00+00:00", state)
    append_risk_event(
        artifacts_dir,
        PaperRiskEvent(
            version_id="paper-test-v1",
            timestamp="2026-03-19T14:06:00+00:00",
            event="stale_closed_bar",
            severity="warning",
            message="latest closed paper bar is stale",
            bar_close_time="2026-03-19T13:00:00+00:00",
        ),
    )
    write_daily_report(
        artifacts_dir,
        session_date="2026-03-19",
        summary=summary,
        state=state,
    )
    return artifacts_dir, deployment_root


def test_load_dashboard_payload_aggregates_latest_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts_dir, deployment_root = write_dashboard_fixture(tmp_path)
    monkeypatch.setattr("paper_trading.dashboard._load_signal_payload", stub_signal_payload)

    payload = load_dashboard_payload(artifacts_dir, deployment_root)

    assert payload["manifest"]["version_id"] == "paper-test-v1"
    assert payload["latest_status"]["status"] == "ok"
    assert payload["stats"]["current_position_side"] == "long"
    assert payload["stats"]["current_notional_usdc"] == pytest.approx(9_000.0)
    assert payload["stats"]["turnover_usdc"] == pytest.approx(9_000.0)
    assert payload["stats"]["fill_count"] == 1
    assert payload["stats"]["risk_event_count"] == 1
    assert payload["equity_points"][0]["net_pnl_usdc"] == pytest.approx(135.0)
    assert payload["recent_decisions"][0]["action"] == "buy"
    assert payload["signal_state"]["available"] is True
    assert payload["recent_decisions"][0]["signal_story"]["headline"] == "Long entry fired"
    assert payload["files"]["decisions"]["row_count"] == 1


def test_dashboard_app_routes_html_and_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts_dir, deployment_root = write_dashboard_fixture(tmp_path)
    monkeypatch.setattr("paper_trading.dashboard._load_signal_payload", stub_signal_payload)
    app = DashboardApp(
        artifacts_dir=artifacts_dir,
        deployment_root=deployment_root,
        refresh_interval_ms=7_000,
    )

    status, content_type, body = app.route("/")
    assert status == HTTPStatus.OK
    assert content_type.startswith("text/html")
    assert "7000" in body.decode("utf-8")

    status, content_type, body = app.route("/api/dashboard")
    assert status == HTTPStatus.OK
    assert content_type.startswith("application/json")
    payload = json.loads(body.decode("utf-8"))
    assert payload["paths"]["artifacts_dir"] == str(artifacts_dir)
    assert payload["health"]["runner_status"] == "ok"

    status, _, body = app.route("/missing")
    assert status == HTTPStatus.NOT_FOUND
    assert body == b"Not found"
