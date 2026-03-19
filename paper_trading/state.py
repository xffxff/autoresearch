from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from paper_trading.types import PaperState


def new_state(version_id: str) -> PaperState:
    return PaperState(version_id=version_id)


def load_state(path: Path) -> PaperState:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return PaperState(**payload)


def save_state(path: Path, state: PaperState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")


def load_or_initialize(path: Path, version_id: str) -> PaperState:
    if not path.exists():
        return new_state(version_id)
    state = load_state(path)
    if state.version_id != version_id:
        return new_state(version_id)
    return state
