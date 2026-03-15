from __future__ import annotations

from datetime import date
from pathlib import Path

from download_data import build_download_plan


def test_build_download_plan_uses_monthly_funding_files_only() -> None:
    plan = build_download_plan(
        root=Path("data/raw"),
        datasets=["fundingRate"],
        symbol="BTCUSDT",
        interval="1h",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 3),
        as_of=date(2024, 2, 10),
    )

    assert len(plan) == 1
    assert plan[0].period_kind == "monthly"
    assert plan[0].remote_path.endswith("BTCUSDT-fundingRate-2024-01.zip")
