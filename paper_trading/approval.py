from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Callable

import pandas as pd

from backtest import MarketBundle
from paper_trading.config import PAPER_SCHEME, PaperConfig
from paper_trading.types import ApprovalManifest, CurrentDeployment


FeatureBuilder = Callable[[MarketBundle], pd.DataFrame]
StrategyBuilder = Callable[[pd.DataFrame, MarketBundle], pd.Series]


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_current_deployment(deployment_root: Path) -> CurrentDeployment:
    payload = json.loads((deployment_root / "current.json").read_text(encoding="utf-8"))
    version_id = str(payload.get("version_id", "")).strip()
    if not version_id:
        raise ValueError("paper current deployment missing version_id")
    return CurrentDeployment(version_id=version_id)


def load_manifest(version_dir: Path) -> ApprovalManifest:
    payload = json.loads((version_dir / "manifest.json").read_text(encoding="utf-8"))
    return ApprovalManifest(**payload)


def validate_manifest(manifest: ApprovalManifest, config: PaperConfig) -> None:
    if manifest.paper_scheme != PAPER_SCHEME:
        raise ValueError(
            f"paper scheme mismatch: expected {PAPER_SCHEME}, got {manifest.paper_scheme}"
        )
    if manifest.symbol != config.symbol:
        raise ValueError(
            f"paper symbol mismatch: expected {config.symbol}, got {manifest.symbol}"
        )
    if manifest.bar_interval != config.bar_interval:
        raise ValueError(
            f"paper interval mismatch: expected {config.bar_interval}, got {manifest.bar_interval}"
        )
    if abs(manifest.capital_usdc - config.capital_usdc) > 1e-9:
        raise ValueError(
            f"paper capital mismatch: expected {config.capital_usdc}, got {manifest.capital_usdc}"
        )
    if manifest.execution_mode != "taker":
        raise ValueError("paper execution mode must be taker")
    if abs(manifest.taker_fee_bps - config.taker_fee_bps) > 1e-9:
        raise ValueError(
            f"paper taker fee mismatch: expected {config.taker_fee_bps}, got {manifest.taker_fee_bps}"
        )
    if abs(manifest.extra_slippage_bps - config.extra_slippage_bps) > 1e-9:
        raise ValueError(
            "paper extra slippage mismatch: "
            f"expected {config.extra_slippage_bps}, got {manifest.extra_slippage_bps}"
        )


def validate_snapshot_hashes(version_dir: Path, manifest: ApprovalManifest) -> None:
    features_hash = compute_sha256(version_dir / "features.py")
    strategy_hash = compute_sha256(version_dir / "strategy.py")
    if features_hash != manifest.features_sha256:
        raise ValueError("approved paper features hash mismatch")
    if strategy_hash != manifest.strategy_sha256:
        raise ValueError("approved paper strategy hash mismatch")


def _load_module(module_path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to import module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_approved_modules(
    deployment_root: Path, config: PaperConfig
) -> tuple[ApprovalManifest, ModuleType, ModuleType]:
    current = load_current_deployment(deployment_root)
    version_dir = deployment_root / "versions" / current.version_id
    manifest = load_manifest(version_dir)
    validate_manifest(manifest, config)
    validate_snapshot_hashes(version_dir, manifest)
    features_module = _load_module(
        version_dir / "features.py", f"paper_features_{manifest.version_id}"
    )
    strategy_module = _load_module(
        version_dir / "strategy.py", f"paper_strategy_{manifest.version_id}"
    )
    return manifest, features_module, strategy_module


def load_approved_builders(
    deployment_root: Path, config: PaperConfig
) -> tuple[ApprovalManifest, FeatureBuilder, StrategyBuilder]:
    manifest, features_module, strategy_module = load_approved_modules(
        deployment_root, config
    )
    feature_builder = getattr(features_module, "build_features", None)
    strategy_builder = getattr(strategy_module, "generate_position", None)
    if feature_builder is None:
        raise AttributeError("approved paper features module missing build_features")
    if strategy_builder is None:
        raise AttributeError("approved paper strategy module missing generate_position")
    return manifest, feature_builder, strategy_builder
