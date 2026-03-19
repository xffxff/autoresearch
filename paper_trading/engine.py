from __future__ import annotations

from datetime import datetime, timezone

from paper_trading.config import PaperConfig
from paper_trading.types import BboSnapshot, PaperDecision, PaperFill, PaperState


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def settle_open_to_open_pnl(state: PaperState, current_open_price: float) -> PaperState:
    if state.last_mark_price is None:
        state.last_mark_price = current_open_price
        return state
    gross_delta = state.current_qty * (current_open_price - state.last_mark_price)
    state.gross_pnl_usdc += gross_delta
    state.net_pnl_usdc += gross_delta
    state.last_mark_price = current_open_price
    return state


def apply_bar_funding(
    state: PaperState, funding_rate: float, reference_price: float
) -> PaperState:
    funding_delta = -state.current_qty * reference_price * funding_rate
    state.funding_pnl_usdc += funding_delta
    state.net_pnl_usdc += funding_delta
    return state


def compute_target_qty(
    target_position: float, capital_usdc: float, fill_price: float
) -> float:
    if abs(target_position) <= 1e-12:
        return 0.0
    return target_position * capital_usdc / fill_price


def choose_fill_price(
    delta_qty: float, bbo: BboSnapshot, extra_slippage_bps: float
) -> tuple[str, float]:
    slippage_rate = extra_slippage_bps / 10_000.0
    if delta_qty > 0.0:
        return "buy", bbo.ask_px * (1.0 + slippage_rate)
    return "sell", bbo.bid_px * (1.0 - slippage_rate)


def build_decision(
    *,
    version_id: str,
    bar_close_time: str,
    execution_bar_time: str,
    target_position: float,
    state: PaperState,
    capital_usdc: float,
    open_price: float,
    bbo: BboSnapshot | None,
    extra_slippage_bps: float,
    reason: str = "ok",
) -> PaperDecision:
    reference_target_qty = compute_target_qty(target_position, capital_usdc, open_price)
    reference_delta_qty = reference_target_qty - state.current_qty
    if abs(reference_delta_qty) <= 1e-12:
        return PaperDecision(
            version_id=version_id,
            bar_close_time=bar_close_time,
            execution_bar_time=execution_bar_time,
            decision_time=now_iso(),
            target_position=target_position,
            current_qty=state.current_qty,
            target_qty=state.current_qty,
            delta_qty=0.0,
            open_price=open_price,
            fill_price=None,
            status="hold",
            action="hold",
            reason=reason,
        )
    if bbo is None:
        return PaperDecision(
            version_id=version_id,
            bar_close_time=bar_close_time,
            execution_bar_time=execution_bar_time,
            decision_time=now_iso(),
            target_position=target_position,
            current_qty=state.current_qty,
            target_qty=state.current_qty,
            delta_qty=0.0,
            open_price=open_price,
            fill_price=None,
            status="skip",
            action="skip",
            reason="missing_bbo",
        )
    side, fill_price = choose_fill_price(reference_delta_qty, bbo, extra_slippage_bps)
    target_qty = compute_target_qty(target_position, capital_usdc, fill_price)
    delta_qty = target_qty - state.current_qty
    action = "hold" if abs(delta_qty) <= 1e-12 else side
    status = "hold" if action == "hold" else "trade"
    return PaperDecision(
        version_id=version_id,
        bar_close_time=bar_close_time,
        execution_bar_time=execution_bar_time,
        decision_time=now_iso(),
        target_position=target_position,
        current_qty=state.current_qty,
        target_qty=target_qty,
        delta_qty=delta_qty,
        open_price=open_price,
        fill_price=fill_price,
        status=status,
        action=action,
        reason=reason,
    )


def execute_rebalance(
    state: PaperState, decision: PaperDecision, config: PaperConfig
) -> tuple[PaperState, PaperFill | None]:
    if decision.status != "trade" or decision.fill_price is None:
        if decision.status != "skip":
            state.last_bar_time = decision.execution_bar_time
        return state, None
    notional_usdc = abs(decision.delta_qty) * decision.fill_price
    fee_usdc = notional_usdc * (config.taker_fee_bps / 10_000.0)
    slippage_usdc = abs(decision.delta_qty) * abs(
        decision.fill_price - decision.open_price
    )
    state.current_qty = decision.target_qty
    state.last_bar_time = decision.execution_bar_time
    state.last_mark_price = decision.open_price
    state.fee_pnl_usdc -= fee_usdc
    state.slippage_pnl_usdc -= slippage_usdc
    state.net_pnl_usdc -= fee_usdc + slippage_usdc
    fill = PaperFill(
        version_id=decision.version_id,
        timestamp=decision.decision_time,
        execution_bar_time=decision.execution_bar_time,
        side=decision.action,
        qty=abs(decision.delta_qty),
        price=decision.fill_price,
        notional_usdc=notional_usdc,
        fee_usdc=fee_usdc,
        slippage_usdc=slippage_usdc,
    )
    return state, fill


def process_bar_roll(
    *,
    state: PaperState,
    version_id: str,
    bar_close_time: str,
    execution_bar_time: str,
    target_position: float,
    current_open_price: float,
    bar_funding_rate: float,
    bbo: BboSnapshot | None,
    config: PaperConfig,
) -> tuple[PaperState, PaperDecision, PaperFill | None]:
    settle_open_to_open_pnl(state, current_open_price)
    apply_bar_funding(state, bar_funding_rate, reference_price=current_open_price)
    decision = build_decision(
        version_id=version_id,
        bar_close_time=bar_close_time,
        execution_bar_time=execution_bar_time,
        target_position=target_position,
        state=state,
        capital_usdc=config.capital_usdc,
        open_price=current_open_price,
        bbo=bbo,
        extra_slippage_bps=config.extra_slippage_bps,
    )
    state, fill = execute_rebalance(state, decision, config)
    return state, decision, fill
