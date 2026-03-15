from __future__ import annotations

import argparse
import calendar
import hashlib
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen


REMOTE_ROOT = "https://data.binance.vision/data/futures/um"
DEFAULT_ROOT = Path("data/raw")
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVAL = "1h"
DEFAULT_START_DATE = date(2020, 1, 1)
KLINE_DATASETS = ("klines", "markPriceKlines", "indexPriceKlines", "premiumIndexKlines")
ALL_DATASETS = KLINE_DATASETS + ("fundingRate",)


@dataclass(frozen=True)
class DownloadItem:
    dataset: str
    period_kind: str
    token: str
    remote_path: str
    local_zip: Path
    local_checksum: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Binance public BTCUSDT USD-M data.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Directory for raw downloaded files.")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Only BTCUSDT is supported in v1.")
    parser.add_argument("--interval", default=DEFAULT_INTERVAL, help="Kline interval. Only 1h is supported in v1.")
    parser.add_argument("--start-date", type=parse_date, default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", type=parse_date, default=date.today())
    parser.add_argument("--datasets", nargs="+", choices=ALL_DATASETS, default=list(ALL_DATASETS))
    parser.add_argument("--force", action="store_true", help="Re-download files even if checksums already match.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned files without downloading.")
    return parser.parse_args()


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def month_floor(day: date) -> date:
    return day.replace(day=1)


def add_months(day: date, months: int) -> date:
    year = day.year + (day.month - 1 + months) // 12
    month = (day.month - 1 + months) % 12 + 1
    return date(year, month, 1)


def iter_month_starts(start_date: date, end_date: date) -> list[date]:
    current = month_floor(start_date)
    months: list[date] = []
    while current <= end_date:
        months.append(current)
        current = add_months(current, 1)
    return months


def iter_days(start_date: date, end_date: date) -> list[date]:
    current = start_date
    days: list[date] = []
    while current <= end_date:
        days.append(current)
        current += timedelta(days=1)
    return days


def first_monday(day: date) -> date:
    offset = (calendar.MONDAY - day.weekday()) % 7
    return day + timedelta(days=offset)


def monthly_available(month_start: date, as_of: date) -> bool:
    next_month_start = add_months(month_start, 1)
    available_on = first_monday(next_month_start)
    return as_of >= available_on


def month_last_day(month_start: date) -> date:
    _, last_day = calendar.monthrange(month_start.year, month_start.month)
    return month_start.replace(day=last_day)


def build_remote_path(
    dataset: str,
    period_kind: str,
    token: str,
    symbol: str,
    interval: str,
    checksum: bool,
) -> str:
    suffix = ".zip.CHECKSUM" if checksum else ".zip"
    if dataset == "fundingRate":
        file_name = f"{symbol}-fundingRate-{token}{suffix}"
        return f"data/futures/um/{period_kind}/{dataset}/{symbol}/{file_name}"
    file_name = f"{symbol}-{interval}-{token}{suffix}"
    return f"data/futures/um/{period_kind}/{dataset}/{symbol}/{interval}/{file_name}"


def build_download_plan(
    root: Path,
    datasets: list[str],
    symbol: str,
    interval: str,
    start_date: date,
    end_date: date,
    as_of: date,
) -> list[DownloadItem]:
    items: list[DownloadItem] = []
    for dataset in datasets:
        for month_start in iter_month_starts(start_date, end_date):
            month_end = month_last_day(month_start)
            if dataset == "fundingRate":
                if monthly_available(month_start, as_of):
                    token = month_start.strftime("%Y-%m")
                    remote_path = build_remote_path(dataset, "monthly", token, symbol, interval, checksum=False)
                    checksum_path = build_remote_path(dataset, "monthly", token, symbol, interval, checksum=True)
                    local_zip = root / remote_path.replace("data/", "", 1)
                    local_checksum = root / checksum_path.replace("data/", "", 1)
                    items.append(
                        DownloadItem(
                            dataset=dataset,
                            period_kind="monthly",
                            token=token,
                            remote_path=remote_path,
                            local_zip=local_zip,
                            local_checksum=local_checksum,
                        )
                    )
                continue

            full_month_requested = start_date <= month_start and month_end <= end_date
            if full_month_requested and monthly_available(month_start, as_of):
                token = month_start.strftime("%Y-%m")
                remote_path = build_remote_path(dataset, "monthly", token, symbol, interval, checksum=False)
                checksum_path = build_remote_path(dataset, "monthly", token, symbol, interval, checksum=True)
                local_zip = root / remote_path.replace("data/", "", 1)
                local_checksum = root / checksum_path.replace("data/", "", 1)
                items.append(
                    DownloadItem(
                        dataset=dataset,
                        period_kind="monthly",
                        token=token,
                        remote_path=remote_path,
                        local_zip=local_zip,
                        local_checksum=local_checksum,
                    )
                )
                continue

            for current_day in iter_days(max(start_date, month_start), min(end_date, month_end)):
                token = current_day.strftime("%Y-%m-%d")
                remote_path = build_remote_path(dataset, "daily", token, symbol, interval, checksum=False)
                checksum_path = build_remote_path(dataset, "daily", token, symbol, interval, checksum=True)
                local_zip = root / remote_path.replace("data/", "", 1)
                local_checksum = root / checksum_path.replace("data/", "", 1)
                items.append(
                    DownloadItem(
                        dataset=dataset,
                        period_kind="daily",
                        token=token,
                        remote_path=remote_path,
                        local_zip=local_zip,
                        local_checksum=local_checksum,
                    )
                )
    return items


def download_bytes(url: str) -> bytes:
    with urlopen(url) as response:
        return response.read()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_checksum(path: Path) -> str:
    contents = path.read_text(encoding="utf-8").strip()
    first_line = contents.splitlines()[0]
    return first_line.split()[0]


def ensure_checksum(item: DownloadItem, force: bool) -> str:
    if not item.local_checksum.exists() or force:
        item.local_checksum.parent.mkdir(parents=True, exist_ok=True)
        checksum_url = f"https://data.binance.vision/{build_remote_path(item.dataset, item.period_kind, item.token, DEFAULT_SYMBOL, DEFAULT_INTERVAL, checksum=True)}"
        item.local_checksum.write_bytes(download_bytes(checksum_url))
    return parse_checksum(item.local_checksum)


def ensure_download(item: DownloadItem, force: bool) -> bool:
    expected_checksum = ensure_checksum(item, force=force)
    if item.local_zip.exists() and not force:
        actual_checksum = sha256_file(item.local_zip)
        if actual_checksum == expected_checksum:
            return False

    item.local_zip.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://data.binance.vision/{item.remote_path}"
    item.local_zip.write_bytes(download_bytes(url))
    actual_checksum = sha256_file(item.local_zip)
    if actual_checksum != expected_checksum:
        raise ValueError(f"Checksum mismatch for {item.local_zip}")
    return True


def main() -> int:
    args = parse_args()
    if args.symbol != DEFAULT_SYMBOL:
        raise ValueError("v1 only supports BTCUSDT.")
    if args.interval != DEFAULT_INTERVAL:
        raise ValueError("v1 only supports 1h klines.")
    if args.start_date > args.end_date:
        raise ValueError("start-date must not be after end-date.")

    plan = build_download_plan(
        root=args.root,
        datasets=args.datasets,
        symbol=args.symbol,
        interval=args.interval,
        start_date=args.start_date,
        end_date=args.end_date,
        as_of=date.today(),
    )
    if args.dry_run:
        for item in plan:
            print(item.local_zip)
        return 0

    downloaded = 0
    skipped = 0
    for item in plan:
        try:
            changed = ensure_download(item, force=args.force)
        except HTTPError as exc:
            print(f"missing remote file: {item.remote_path} ({exc.code})", file=sys.stderr)
            continue
        if changed:
            downloaded += 1
            print(f"downloaded {item.local_zip}")
        else:
            skipped += 1
            print(f"verified {item.local_zip}")

    print("---")
    print(f"downloaded: {downloaded}")
    print(f"verified:   {skipped}")
    print(f"total:      {len(plan)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
