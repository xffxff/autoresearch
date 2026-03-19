from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PAPER_SCHEME = "hyperliquid_mainnet_paper_v1"
DEFAULT_MAINNET_API_URL = "https://api.hyperliquid.xyz"


@dataclass(frozen=True)
class PaperConfig:
    symbol: str = "BTC"
    bar_interval: str = "1h"
    capital_usdc: float = 10_000.0
    taker_fee_bps: float = 5.0
    extra_slippage_bps: float = 1.0
    warmup_bars: int = 24 * 120
    max_backfill_bars: int = 5_000
    stale_quote_seconds: int = 30
    max_closed_bar_age_seconds: int = 2 * 3600
    deployment_root: Path = Path("deployments/paper")
    artifacts_dir: Path = Path("paper_artifacts")
    base_url: str = DEFAULT_MAINNET_API_URL

    @property
    def halt_file(self) -> Path:
        return self.artifacts_dir / "HALT"


def default_config() -> PaperConfig:
    return PaperConfig()
