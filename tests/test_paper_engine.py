from __future__ import annotations

import pytest

from paper_trading.config import PaperConfig
from paper_trading.engine import build_decision, process_bar_roll
from paper_trading.state import new_state
from paper_trading.types import BboSnapshot


def test_process_bar_roll_executes_taker_trade_and_tracks_costs() -> None:
    config = PaperConfig(
        capital_usdc=10_000.0, taker_fee_bps=5.0, extra_slippage_bps=1.0
    )
    state = new_state("v1")
    bbo = BboSnapshot(
        timestamp="2026-03-18T01:00:05+00:00", bid_px=99.95, ask_px=100.05
    )

    state, decision, fill = process_bar_roll(
        state=state,
        version_id="v1",
        bar_close_time="2026-03-18T00:00:00+00:00",
        execution_bar_time="2026-03-18T01:00:00+00:00",
        target_position=1.0,
        current_open_price=100.0,
        bar_funding_rate=0.0,
        bbo=bbo,
        config=config,
    )

    expected_price = 100.05 * (1.0 + config.extra_slippage_bps / 10_000.0)
    expected_qty = config.capital_usdc / expected_price
    expected_fee = expected_qty * expected_price * (config.taker_fee_bps / 10_000.0)
    expected_slippage = expected_qty * abs(expected_price - 100.0)

    assert decision.action == "buy"
    assert fill is not None
    assert fill.price == pytest.approx(expected_price)
    assert state.current_qty == pytest.approx(expected_qty)
    assert state.fee_pnl_usdc == pytest.approx(-expected_fee)
    assert state.slippage_pnl_usdc == pytest.approx(-expected_slippage)
    assert state.net_pnl_usdc == pytest.approx(-(expected_fee + expected_slippage))


def test_build_decision_uses_fixed_capital_not_equity() -> None:
    config = PaperConfig(capital_usdc=10_000.0)
    state = new_state("v1")
    state.net_pnl_usdc = 2_500.0
    bbo = BboSnapshot(timestamp="2026-03-18T01:00:05+00:00", bid_px=199.8, ask_px=200.2)

    decision = build_decision(
        version_id="v1",
        bar_close_time="2026-03-18T00:00:00+00:00",
        execution_bar_time="2026-03-18T01:00:00+00:00",
        target_position=0.5,
        state=state,
        capital_usdc=config.capital_usdc,
        open_price=200.0,
        bbo=bbo,
        extra_slippage_bps=config.extra_slippage_bps,
    )

    expected_fill = 200.2 * (1.0 + config.extra_slippage_bps / 10_000.0)
    assert decision.target_qty == pytest.approx(
        (config.capital_usdc * 0.5) / expected_fill
    )


def test_build_decision_skips_when_bbo_missing() -> None:
    decision = build_decision(
        version_id="v1",
        bar_close_time="2026-03-18T00:00:00+00:00",
        execution_bar_time="2026-03-18T01:00:00+00:00",
        target_position=1.0,
        state=new_state("v1"),
        capital_usdc=10_000.0,
        open_price=100.0,
        bbo=None,
        extra_slippage_bps=1.0,
    )

    assert decision.status == "skip"
    assert decision.reason == "missing_bbo"
