from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from paper_trading.approval import load_approved_builders
from paper_trading.config import PaperConfig
from tests.paper_helpers import write_approved_snapshot


def make_frame() -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=5, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "volume": [10.0] * 5,
            "mark_close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "index_close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "premium_close": [0.0] * 5,
            "funding_rate": [0.0] * 5,
        },
        index=index,
    )


def test_load_approved_builders_reads_snapshot(tmp_path: Path) -> None:
    config = PaperConfig(deployment_root=tmp_path / "deployments" / "paper")
    write_approved_snapshot(config.deployment_root, config)

    manifest, feature_builder, strategy_builder = load_approved_builders(
        config.deployment_root, config
    )

    market_frame = make_frame()
    features = feature_builder(
        type("Bundle", (), {"frame": market_frame, "index": market_frame.index})()
    )
    positions = strategy_builder(
        features, type("Bundle", (), {"frame": market_frame})()
    )

    assert manifest.version_id == "paper-test-v1"
    assert list(features.columns) == ["signal"]
    assert (positions == 1.0).all()


def test_load_approved_builders_rejects_capital_mismatch(tmp_path: Path) -> None:
    config = PaperConfig(deployment_root=tmp_path / "deployments" / "paper")
    write_approved_snapshot(config.deployment_root, config)
    bad_config = PaperConfig(
        deployment_root=config.deployment_root, capital_usdc=20_000.0
    )

    with pytest.raises(ValueError, match="paper capital mismatch"):
        load_approved_builders(config.deployment_root, bad_config)
