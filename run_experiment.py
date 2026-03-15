from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from backtest import BacktestResult, EvaluationConfig, MarketBundle, evaluate_strategy
from features import build_features
from strategy import generate_position


DEFAULT_DATASET = Path("data/derived/btcusdt_um_1h.parquet")
DEFAULT_ARTIFACTS = Path("artifacts")
RESULTS_HEADER = [
    "commit",
    "score",
    "net_sharpe",
    "net_return",
    "max_drawdown",
    "trade_count",
    "turnover",
    "active_windows",
    "worst_window_return",
    "pass_gates",
    "status",
    "description",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fixed BTC strategy walk-forward experiment.")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--results-file", type=Path, help="Append the result to a TSV file.")
    parser.add_argument("--description", default="manual")
    parser.add_argument("--commit", help="Override the commit hash written to results.tsv.")
    parser.add_argument("--incumbent-score", type=float, help="Override the score threshold used for keep/discard.")
    return parser.parse_args()


def load_market_bundle(dataset_path: Path) -> MarketBundle:
    frame = pd.read_parquet(dataset_path)
    frame.index = pd.DatetimeIndex(frame.index).tz_convert("UTC") if frame.index.tz is not None else pd.DatetimeIndex(frame.index).tz_localize("UTC")
    return MarketBundle(frame=frame.sort_index())


def get_git_commit() -> str:
    try:
        output = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        return output or "worktree"
    except Exception:
        return "worktree"


def ensure_results_header(path: Path) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(RESULTS_HEADER)
        return

    frame = pd.read_csv(path, sep="\t")
    for column in RESULTS_HEADER:
        if column not in frame.columns:
            frame[column] = ""
    if "net_return" in frame.columns:
        frame["score"] = frame["net_return"]
    frame = frame.reindex(columns=RESULTS_HEADER)
    frame.to_csv(path, sep="\t", index=False)


def read_best_score(path: Path) -> float | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path, sep="\t")
    if frame.empty:
        return None
    kept = frame[frame["status"] == "keep"]
    if kept.empty:
        return None
    return float(kept["score"].max())


def determine_status(result: BacktestResult, incumbent_score: float | None) -> str:
    if not result.pass_gates:
        return "discard"
    if incumbent_score is None:
        return "keep"
    return "keep" if result.score > incumbent_score else "discard"


def append_result_row(
    path: Path,
    result: BacktestResult,
    description: str,
    commit: str,
    status: str,
) -> None:
    ensure_results_header(path)
    row = {
        "commit": commit,
        "score": f"{result.score:.6f}",
        "net_sharpe": f"{result.net_sharpe:.6f}",
        "net_return": f"{result.net_return:.6f}",
        "max_drawdown": f"{result.max_drawdown:.6f}",
        "trade_count": str(result.trade_count),
        "turnover": f"{result.turnover:.6f}",
        "active_windows": str(result.active_windows),
        "worst_window_return": f"{result.worst_window_return:.6f}",
        "pass_gates": "true" if result.pass_gates else "false",
        "status": status,
        "description": description,
    }
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULTS_HEADER, delimiter="\t")
        writer.writerow(row)


def save_artifacts(artifacts_dir: Path, result: BacktestResult, config: EvaluationConfig) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "score": result.score,
        "net_sharpe": result.net_sharpe,
        "net_return": result.net_return,
        "max_drawdown": result.max_drawdown,
        "trade_count": result.trade_count,
        "turnover": result.turnover,
        "active_windows": result.active_windows,
        "worst_window_return": result.worst_window_return,
        "best_window_return": result.best_window_return,
        "pass_gates": result.pass_gates,
        "config": asdict(config),
        "windows": [
            {
                "calibration_start": window.calibration_start.isoformat(),
                "validation_start": window.validation_start.isoformat(),
                "validation_end": window.validation_end.isoformat(),
                "net_sharpe": window.net_sharpe,
                "net_return": window.net_return,
                "max_drawdown": window.max_drawdown,
                "trade_count": window.trade_count,
                "turnover": window.turnover,
                "bars": window.bars,
            }
            for window in result.windows
        ],
    }
    (artifacts_dir / "latest_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    window_frame = pd.DataFrame(summary_payload["windows"])
    window_frame.to_csv(artifacts_dir / "latest_windows.tsv", sep="\t", index=False)


def render_summary(result: BacktestResult, status: str | None = None) -> str:
    lines = [
        "---",
        f"score:          {result.score:.6f}",
        f"net_sharpe:     {result.net_sharpe:.6f}",
        f"net_return:     {result.net_return:.6f}",
        f"max_drawdown:   {result.max_drawdown:.6f}",
        f"trade_count:    {result.trade_count}",
        f"turnover:       {result.turnover:.6f}",
        f"active_windows: {result.active_windows}",
        f"worst_window_return: {result.worst_window_return:.6f}",
        f"pass_gates:     {'true' if result.pass_gates else 'false'}",
        f"num_windows:    {len(result.windows)}",
    ]
    if status is not None:
        lines.append(f"status:         {status}")
    return "\n".join(lines)


def run_once(
    dataset_path: Path,
    artifacts_dir: Path,
    results_file: Path | None = None,
    description: str = "manual",
    commit: str | None = None,
    incumbent_score: float | None = None,
) -> tuple[BacktestResult, str | None]:
    market = load_market_bundle(dataset_path)
    config = EvaluationConfig()
    result = evaluate_strategy(market, build_features, generate_position, config=config)
    save_artifacts(artifacts_dir, result, config)

    status: str | None = None
    if results_file is not None:
        ensure_results_header(results_file)
        current_commit = commit or get_git_commit()
        effective_incumbent = incumbent_score if incumbent_score is not None else read_best_score(results_file)
        status = determine_status(result, effective_incumbent)
        append_result_row(results_file, result, description=description, commit=current_commit, status=status)
    return result, status


def main() -> int:
    args = parse_args()
    result, status = run_once(
        dataset_path=args.dataset_path,
        artifacts_dir=args.artifacts_dir,
        results_file=args.results_file,
        description=args.description,
        commit=args.commit,
        incumbent_score=args.incumbent_score,
    )
    print(render_summary(result, status=status))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
