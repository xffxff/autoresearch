from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_RAW_ROOT = Path("data/raw")
DEFAULT_OUTPUT_PATH = Path("data/derived/btcusdt_um_1h.parquet")
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVAL = "1h"
PRICE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]
FUNDING_COLUMNS = ["calc_time", "funding_interval_hours", "funding_rate"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a normalized BTCUSDT USD-M parquet dataset.")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--interval", default=DEFAULT_INTERVAL)
    return parser.parse_args()


def read_binance_zip_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, compression="zip", header=None)
    first_value = str(frame.iloc[0, 0]).strip().lower()
    if first_value in {"open_time", "calc_time"}:
        frame = frame.iloc[1:].reset_index(drop=True)
    frame.columns = columns
    return frame


def dataset_dir(raw_root: Path, dataset: str, period_kind: str, symbol: str, interval: str) -> Path:
    if dataset == "fundingRate":
        return raw_root / "futures" / "um" / period_kind / dataset / symbol
    return raw_root / "futures" / "um" / period_kind / dataset / symbol / interval


def collect_zip_files(raw_root: Path, dataset: str, symbol: str, interval: str) -> list[Path]:
    files: list[Path] = []
    for period_kind in ("monthly", "daily"):
        directory = dataset_dir(raw_root, dataset, period_kind, symbol, interval)
        if directory.exists():
            files.extend(sorted(directory.glob("*.zip")))
    return files


def load_price_dataset(raw_root: Path, dataset: str, symbol: str, interval: str) -> pd.DataFrame:
    frames = [read_binance_zip_csv(path, PRICE_COLUMNS) for path in collect_zip_files(raw_root, dataset, symbol, interval)]
    if not frames:
        raise FileNotFoundError(f"No raw files found for {dataset}.")

    combined = pd.concat(frames, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["open_time"].astype("int64"), unit="ms", utc=True)
    for column in ("open", "high", "low", "close", "volume"):
        combined[column] = combined[column].astype(float)
    combined = combined.drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp")
    return combined.set_index("timestamp")[["open", "high", "low", "close", "volume"]]


def load_close_series(raw_root: Path, dataset: str, symbol: str, interval: str, target_name: str) -> pd.Series:
    frame = load_price_dataset(raw_root, dataset, symbol, interval)
    return frame["close"].rename(target_name)


def load_funding_dataset(raw_root: Path, symbol: str) -> pd.DataFrame:
    frames = [read_binance_zip_csv(path, FUNDING_COLUMNS) for path in collect_zip_files(raw_root, "fundingRate", symbol, DEFAULT_INTERVAL)]
    if not frames:
        raise FileNotFoundError("No raw files found for fundingRate.")

    combined = pd.concat(frames, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["calc_time"].astype("int64"), unit="ms", utc=True)
    combined["funding_rate"] = combined["funding_rate"].astype(float)
    combined["funding_interval_hours"] = combined["funding_interval_hours"].astype(int)
    combined = combined.drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp")
    return combined.set_index("timestamp")[["funding_interval_hours", "funding_rate"]]


def build_market_frame(raw_root: Path, symbol: str = DEFAULT_SYMBOL, interval: str = DEFAULT_INTERVAL) -> pd.DataFrame:
    base = load_price_dataset(raw_root, "klines", symbol, interval)
    mark = load_close_series(raw_root, "markPriceKlines", symbol, interval, "mark_close")
    index_price = load_close_series(raw_root, "indexPriceKlines", symbol, interval, "index_close")
    premium = load_close_series(raw_root, "premiumIndexKlines", symbol, interval, "premium_close")
    funding = load_funding_dataset(raw_root, symbol)

    frame = base.join(mark, how="left").join(index_price, how="left").join(premium, how="left").join(funding, how="left")
    frame["mark_close"] = frame["mark_close"].ffill().fillna(frame["close"])
    frame["index_close"] = frame["index_close"].ffill().fillna(frame["close"])
    frame["premium_close"] = frame["premium_close"].fillna(0.0)
    frame["funding_interval_hours"] = frame["funding_interval_hours"].fillna(8).astype(int)
    frame["funding_rate"] = frame["funding_rate"].fillna(0.0)
    return frame.sort_index()


def write_market_parquet(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path)


def main() -> int:
    args = parse_args()
    if args.symbol != DEFAULT_SYMBOL:
        raise ValueError("v1 only supports BTCUSDT.")
    if args.interval != DEFAULT_INTERVAL:
        raise ValueError("v1 only supports 1h.")

    frame = build_market_frame(args.raw_root, symbol=args.symbol, interval=args.interval)
    write_market_parquet(frame, args.output_path)

    print("---")
    print(f"rows:        {len(frame)}")
    print(f"start:       {frame.index.min().isoformat()}")
    print(f"end:         {frame.index.max().isoformat()}")
    print(f"output_path: {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

