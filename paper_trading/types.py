from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class CurrentDeployment:
    version_id: str


@dataclass(frozen=True)
class ApprovalManifest:
    version_id: str
    approved_at: str
    approved_by: str
    source_commit: str
    source_research_scheme: str
    source_description: str
    paper_scheme: str
    symbol: str
    bar_interval: str
    capital_usdc: float
    execution_mode: str
    taker_fee_bps: float
    extra_slippage_bps: float
    features_sha256: str
    strategy_sha256: str


@dataclass(frozen=True)
class BboSnapshot:
    timestamp: str
    bid_px: float
    ask_px: float


@dataclass(frozen=True)
class AssetContextSnapshot:
    timestamp: str
    mark_px: float
    oracle_px: float
    premium: float
    funding_rate: float


@dataclass(frozen=True)
class PaperDecision:
    version_id: str
    bar_close_time: str
    execution_bar_time: str
    decision_time: str
    target_position: float
    current_qty: float
    target_qty: float
    delta_qty: float
    open_price: float
    fill_price: float | None
    status: str
    action: str
    reason: str


@dataclass(frozen=True)
class PaperFill:
    version_id: str
    timestamp: str
    execution_bar_time: str
    side: str
    qty: float
    price: float
    notional_usdc: float
    fee_usdc: float
    slippage_usdc: float


@dataclass(frozen=True)
class PaperRiskEvent:
    version_id: str
    timestamp: str
    event: str
    severity: str
    message: str
    bar_close_time: str | None = None


@dataclass
class PaperState:
    version_id: str
    current_qty: float = 0.0
    last_bar_time: str | None = None
    last_mark_price: float | None = None
    gross_pnl_usdc: float = 0.0
    funding_pnl_usdc: float = 0.0
    fee_pnl_usdc: float = 0.0
    slippage_pnl_usdc: float = 0.0
    net_pnl_usdc: float = 0.0


@dataclass(frozen=True)
class PaperRunSummary:
    version_id: str
    status: str
    message: str
    bar_close_time: str | None
    current_qty: float
    gross_pnl_usdc: float
    funding_pnl_usdc: float
    fee_pnl_usdc: float
    slippage_pnl_usdc: float
    net_pnl_usdc: float


def dataclass_dict(value: Any) -> dict[str, Any]:
    return asdict(value)
