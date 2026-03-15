from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from build_dataset import build_market_frame


def write_zip_csv(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    inner_name = path.stem + ".csv"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(inner_name, contents)


def make_price_csv(rows: list[tuple[int, float, float, float, float, float]]) -> str:
    header = "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore\n"
    body = []
    for timestamp, open_, high, low, close, volume in rows:
        body.append(
            f"{timestamp},{open_},{high},{low},{close},{volume},{timestamp + 3599999},0,1,0,0,0"
        )
    return header + "\n".join(body) + "\n"


def make_funding_csv(rows: list[tuple[int, int, float]]) -> str:
    header = "calc_time,funding_interval_hours,funding_rate\n"
    body = [f"{timestamp},{interval},{rate}" for timestamp, interval, rate in rows]
    return header + "\n".join(body) + "\n"


def seed_fixture_tree(root: Path) -> None:
    ts0 = 1704067200000
    ts1 = 1704070800000
    ts2 = 1704074400000

    monthly_rows = [
        (ts0, 100.0, 101.0, 99.0, 100.5, 10.0),
        (ts1, 100.5, 102.0, 100.0, 101.0, 11.0),
    ]
    daily_rows = [
        (ts1, 100.5, 103.0, 100.0, 101.5, 12.0),
        (ts2, 101.5, 104.0, 101.0, 103.0, 13.0),
    ]

    for dataset, monthly_shift, daily_shift in (
        ("klines", 0.0, 0.0),
        ("markPriceKlines", -0.1, -0.1),
        ("indexPriceKlines", -0.2, -0.2),
        ("premiumIndexKlines", 0.01, 0.02),
    ):
        monthly_csv = make_price_csv(
            [(ts, o + monthly_shift, h + monthly_shift, l + monthly_shift, c + monthly_shift, v) for ts, o, h, l, c, v in monthly_rows]
        )
        daily_csv = make_price_csv(
            [(ts, o + daily_shift, h + daily_shift, l + daily_shift, c + daily_shift, v) for ts, o, h, l, c, v in daily_rows]
        )
        write_zip_csv(root / "futures" / "um" / "monthly" / dataset / "BTCUSDT" / "1h" / f"BTCUSDT-1h-2024-01.zip", monthly_csv)
        write_zip_csv(root / "futures" / "um" / "daily" / dataset / "BTCUSDT" / "1h" / f"BTCUSDT-1h-2024-01-02.zip", daily_csv)

    funding_csv = make_funding_csv(
        [
            (ts0, 8, 0.0010),
            (ts2, 8, -0.0005),
        ]
    )
    write_zip_csv(root / "futures" / "um" / "monthly" / "fundingRate" / "BTCUSDT" / "BTCUSDT-fundingRate-2024-01.zip", funding_csv)


def test_build_market_frame_deduplicates_and_aligns(tmp_path: Path) -> None:
    seed_fixture_tree(tmp_path)

    frame = build_market_frame(tmp_path)

    assert list(frame.columns) == [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "mark_close",
        "index_close",
        "premium_close",
        "funding_interval_hours",
        "funding_rate",
    ]
    assert len(frame) == 3
    assert frame.iloc[1]["close"] == 101.5
    assert frame.iloc[0]["funding_rate"] == 0.001
    assert frame.iloc[1]["funding_rate"] == 0.0
    assert frame.iloc[2]["funding_rate"] == -0.0005
    assert frame.index.tz is not None

