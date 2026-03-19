from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from paper_trading.config import PaperConfig
from paper_trading.snapshot import (
    activate_version,
    build_manifest,
    build_version_id,
    copy_strategy_snapshot,
    write_manifest,
)


def test_approve_helpers_copy_snapshot_and_activate(tmp_path: Path) -> None:
    source_features = tmp_path / "features.py"
    source_strategy = tmp_path / "strategy.py"
    source_features.write_text(
        "def build_features(market):\n    return market.frame\n", encoding="utf-8"
    )
    source_strategy.write_text(
        "def generate_position(features, market):\n    return features.iloc[:, 0]\n",
        encoding="utf-8",
    )
    deployment_root = tmp_path / "deployments" / "paper"
    version_dir = deployment_root / "versions" / "v1"

    copy_strategy_snapshot(version_dir, source_features, source_strategy)
    manifest = build_manifest(
        version_id="v1",
        approved_by="fan",
        source_commit="abc1234",
        description="paper approval",
        config=PaperConfig(deployment_root=deployment_root),
        version_dir=version_dir,
    )
    write_manifest(version_dir, manifest)
    activate_version(deployment_root, "v1")

    assert (version_dir / "features.py").read_text(
        encoding="utf-8"
    ) == source_features.read_text(encoding="utf-8")
    assert (version_dir / "strategy.py").read_text(
        encoding="utf-8"
    ) == source_strategy.read_text(encoding="utf-8")
    current = json.loads((deployment_root / "current.json").read_text(encoding="utf-8"))
    stored_manifest = json.loads(
        (version_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert current["version_id"] == "v1"
    assert stored_manifest["approved_by"] == "fan"
    assert stored_manifest["source_commit"] == "abc1234"


def test_build_version_id_includes_time_and_commit() -> None:
    approved_at = datetime(2026, 3, 18, 9, 30, 0, tzinfo=timezone.utc)
    assert build_version_id("abc1234", approved_at) == "20260318-093000_abc1234"
