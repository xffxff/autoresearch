from __future__ import annotations

from pathlib import Path

import pandas as pd

from paper_trading.config import PaperConfig
from paper_trading.types import BboSnapshot, PaperState


def ensure_single_execution(
    state: PaperState, execution_bar_time: pd.Timestamp
) -> None:
    if state.last_bar_time == execution_bar_time.isoformat():
        raise ValueError("paper bar already processed")


def quote_is_fresh(
    bbo: BboSnapshot | None, now: pd.Timestamp, config: PaperConfig
) -> bool:
    if bbo is None:
        return False
    age_seconds = (now - pd.Timestamp(bbo.timestamp)).total_seconds()
    return age_seconds <= config.stale_quote_seconds


def closed_bar_is_recent(
    bar_close_time: pd.Timestamp, now: pd.Timestamp, config: PaperConfig
) -> bool:
    age_seconds = (now - bar_close_time).total_seconds()
    return age_seconds <= config.max_closed_bar_age_seconds


def halt_requested(config: PaperConfig, artifacts_dir: Path | None = None) -> bool:
    halt_file = (artifacts_dir or config.artifacts_dir) / "HALT"
    return halt_file.exists()
