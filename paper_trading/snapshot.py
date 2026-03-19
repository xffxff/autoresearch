from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from paper_trading.approval import compute_sha256
from paper_trading.config import PAPER_SCHEME, PaperConfig, default_config
from paper_trading.types import ApprovalManifest, CurrentDeployment
from run_experiment import RESULTS_SCHEME


DEFAULT_APPROVED_BY = "manual"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Approve a paper-trading strategy snapshot."
    )
    parser.add_argument("--approved-by", default=DEFAULT_APPROVED_BY)
    parser.add_argument("--description", default="manual paper approval")
    parser.add_argument("--source-commit")
    parser.add_argument("--deployment-root", type=Path)
    return parser.parse_args()


def get_git_commit() -> str:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
        return output or "worktree"
    except Exception:
        return "worktree"


def build_version_id(source_commit: str, approved_at: datetime) -> str:
    return f"{approved_at.strftime('%Y%m%d-%H%M%S')}_{source_commit}"


def copy_strategy_snapshot(
    version_dir: Path, source_features: Path, source_strategy: Path
) -> None:
    version_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_features, version_dir / "features.py")
    shutil.copy2(source_strategy, version_dir / "strategy.py")


def build_manifest(
    *,
    version_id: str,
    approved_by: str,
    source_commit: str,
    description: str,
    config: PaperConfig,
    version_dir: Path,
) -> ApprovalManifest:
    return ApprovalManifest(
        version_id=version_id,
        approved_at=datetime.now(timezone.utc).isoformat(),
        approved_by=approved_by,
        source_commit=source_commit,
        source_research_scheme=RESULTS_SCHEME,
        source_description=description,
        paper_scheme=PAPER_SCHEME,
        symbol=config.symbol,
        bar_interval=config.bar_interval,
        capital_usdc=config.capital_usdc,
        execution_mode="taker",
        taker_fee_bps=config.taker_fee_bps,
        extra_slippage_bps=config.extra_slippage_bps,
        features_sha256=compute_sha256(version_dir / "features.py"),
        strategy_sha256=compute_sha256(version_dir / "strategy.py"),
    )


def write_manifest(version_dir: Path, manifest: ApprovalManifest) -> None:
    (version_dir / "manifest.json").write_text(
        json.dumps(asdict(manifest), indent=2), encoding="utf-8"
    )


def activate_version(deployment_root: Path, version_id: str) -> None:
    deployment_root.mkdir(parents=True, exist_ok=True)
    payload = CurrentDeployment(version_id=version_id)
    (deployment_root / "current.json").write_text(
        json.dumps(asdict(payload), indent=2), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    config = default_config()
    deployment_root = args.deployment_root or config.deployment_root
    source_commit = args.source_commit or get_git_commit()
    approved_at = datetime.now(timezone.utc)
    version_id = build_version_id(source_commit, approved_at)
    version_dir = deployment_root / "versions" / version_id
    copy_strategy_snapshot(version_dir, Path("features.py"), Path("strategy.py"))
    manifest = build_manifest(
        version_id=version_id,
        approved_by=args.approved_by,
        source_commit=source_commit,
        description=args.description,
        config=config,
        version_dir=version_dir,
    )
    write_manifest(version_dir, manifest)
    activate_version(deployment_root, version_id)
    print("---")
    print(f"version_id:    {version_id}")
    print(f"approved_by:   {manifest.approved_by}")
    print(f"source_commit: {manifest.source_commit}")
    print(f"deployment:    {deployment_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
