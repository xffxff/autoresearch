from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from paper_trading.approval import compute_sha256
from paper_trading.config import PAPER_SCHEME, PaperConfig
from paper_trading.types import ApprovalManifest, CurrentDeployment


FEATURES_SOURCE = """from __future__ import annotations

import pandas as pd

from backtest import MarketBundle


def build_features(market: MarketBundle) -> pd.DataFrame:
    return pd.DataFrame({"signal": market.frame["close"].pct_change().fillna(0.0)}, index=market.frame.index)
"""


STRATEGY_SOURCE = """from __future__ import annotations

import pandas as pd

from backtest import MarketBundle


def generate_position(features: pd.DataFrame, market: MarketBundle) -> pd.Series:
    return pd.Series(1.0, index=features.index, name="position")
"""


def write_approved_snapshot(
    deployment_root: Path,
    config: PaperConfig,
    *,
    version_id: str = "paper-test-v1",
) -> Path:
    version_dir = deployment_root / "versions" / version_id
    version_dir.mkdir(parents=True, exist_ok=True)
    features_path = version_dir / "features.py"
    strategy_path = version_dir / "strategy.py"
    features_path.write_text(FEATURES_SOURCE, encoding="utf-8")
    strategy_path.write_text(STRATEGY_SOURCE, encoding="utf-8")
    manifest = ApprovalManifest(
        version_id=version_id,
        approved_at="2026-03-18T00:00:00+00:00",
        approved_by="test",
        source_commit="abc1234",
        source_research_scheme="research_holdout_v2_8windows",
        source_description="test snapshot",
        paper_scheme=PAPER_SCHEME,
        symbol=config.symbol,
        bar_interval=config.bar_interval,
        capital_usdc=config.capital_usdc,
        execution_mode="taker",
        taker_fee_bps=config.taker_fee_bps,
        extra_slippage_bps=config.extra_slippage_bps,
        features_sha256=compute_sha256(features_path),
        strategy_sha256=compute_sha256(strategy_path),
    )
    (version_dir / "manifest.json").write_text(
        json.dumps(asdict(manifest), indent=2), encoding="utf-8"
    )
    deployment_root.mkdir(parents=True, exist_ok=True)
    (deployment_root / "current.json").write_text(
        json.dumps(asdict(CurrentDeployment(version_id=version_id)), indent=2),
        encoding="utf-8",
    )
    return version_dir


def make_candles(index: pd.DatetimeIndex, *, closed: bool) -> pd.DataFrame:
    values = pd.Series(range(len(index)), index=index, dtype=float)
    frame = pd.DataFrame(
        {
            "open": 100.0 + values,
            "high": 100.5 + values,
            "low": 99.5 + values,
            "close": 100.25 + values,
            "volume": 10.0 + values,
            "close_time": index + pd.Timedelta(hours=1),
            "is_closed": closed,
        },
        index=index,
    )
    return frame
