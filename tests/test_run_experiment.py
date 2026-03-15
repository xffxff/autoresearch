from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from backtest import EvaluationConfig
from run_experiment import RESULTS_SCHEME, determine_status, read_best_score, render_summary, run_once


def make_dataset(path: Path, length: int = 24 * 1_200) -> None:
    index = pd.date_range("2021-01-01", periods=length, freq="1h", tz="UTC")
    phase = np.linspace(0.0, 80.0 * np.pi, length)
    drift = 0.0006 * np.sin(phase) + 0.00005
    open_ = 20_000 * np.cumprod(1.0 + drift)
    frame = pd.DataFrame(
        {
            "open": open_,
            "high": open_ * 1.001,
            "low": open_ * 0.999,
            "close": open_ * (1.0 + drift / 2.0),
            "volume": np.full(length, 100.0),
            "mark_close": open_ * (1.0 + drift / 2.0),
            "index_close": open_ * (1.0 + drift / 3.0),
            "premium_close": 0.0001 * np.sin(phase / 6.0),
            "funding_interval_hours": np.full(length, 8),
            "funding_rate": np.where(np.arange(length) % 8 == 0, 0.00005 * np.sin(phase), 0.0),
        },
        index=index,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)


def test_run_once_appends_results_and_writes_artifacts(tmp_path: Path) -> None:
    dataset_path = tmp_path / "btc.parquet"
    results_path = tmp_path / "results.tsv"
    artifacts_dir = tmp_path / "artifacts"
    make_dataset(dataset_path)

    result, status = run_once(
        dataset_path=dataset_path,
        artifacts_dir=artifacts_dir,
        results_file=results_path,
        description="baseline",
        commit="abc1234",
    )

    assert status == "keep"
    assert results_path.exists()
    assert artifacts_dir.joinpath("latest_summary.json").exists()
    assert artifacts_dir.joinpath("latest_windows.tsv").exists()
    assert read_best_score(results_path, RESULTS_SCHEME) == pytest.approx(result.score, rel=0, abs=1e-6)
    assert "score:" in render_summary(result, status=status)
    assert "active_windows:" in render_summary(result, status=status)
    assert "worst_window_return:" in render_summary(result, status=status)
    assert determine_status(result, incumbent_score=result.score + 1.0) == "discard"

    recorded = pd.read_csv(results_path, sep="\t")
    assert "scheme" in recorded.columns
    assert "active_windows" in recorded.columns
    assert "worst_window_return" in recorded.columns
    assert recorded.iloc[0]["score"] == pytest.approx(result.net_return, abs=1e-6)
    assert recorded.iloc[0]["scheme"] == RESULTS_SCHEME

    summary = json.loads((artifacts_dir / "latest_summary.json").read_text(encoding="utf-8"))
    assert summary["scheme"] == RESULTS_SCHEME
    assert summary["active_windows"] == result.active_windows
    assert summary["worst_window_return"] == pytest.approx(result.worst_window_return)
    assert summary["best_window_return"] == pytest.approx(result.best_window_return)


def test_run_once_migrates_old_results_header(tmp_path: Path) -> None:
    dataset_path = tmp_path / "btc.parquet"
    results_path = tmp_path / "results.tsv"
    artifacts_dir = tmp_path / "artifacts"
    make_dataset(dataset_path)
    results_path.write_text(
        "commit\tscore\tnet_sharpe\tnet_return\tmax_drawdown\ttrade_count\tturnover\tpass_gates\tstatus\tdescription\n"
        "old1234\t1.500000\t0.800000\t0.250000\t0.100000\t40\t20.000000\ttrue\tkeep\told row\n",
        encoding="utf-8",
    )

    run_once(
        dataset_path=dataset_path,
        artifacts_dir=artifacts_dir,
        results_file=results_path,
        description="after migration",
        commit="new1234",
    )

    recorded = pd.read_csv(results_path, sep="\t")
    assert list(recorded.columns) == [
        "commit",
        "scheme",
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
    assert recorded.iloc[0]["score"] == pytest.approx(0.25, abs=1e-6)
    assert pd.isna(recorded.iloc[0]["scheme"]) or recorded.iloc[0]["scheme"] == ""


def test_run_once_ignores_legacy_rows_when_computing_incumbent(tmp_path: Path) -> None:
    dataset_path = tmp_path / "btc.parquet"
    results_path = tmp_path / "results.tsv"
    artifacts_dir = tmp_path / "artifacts"
    make_dataset(dataset_path)
    results_path.write_text(
        "commit\tscore\tnet_sharpe\tnet_return\tmax_drawdown\ttrade_count\tturnover\tpass_gates\tstatus\tdescription\n"
        "legacy1\t9.999999\t0.800000\t9.999999\t0.100000\t40\t20.000000\ttrue\tkeep\tlegacy row\n",
        encoding="utf-8",
    )

    _, status = run_once(
        dataset_path=dataset_path,
        artifacts_dir=artifacts_dir,
        results_file=results_path,
        description="new scheme baseline",
        commit="fresh001",
    )

    assert status == "keep"


def test_run_once_can_include_holdout_summary(tmp_path: Path) -> None:
    dataset_path = tmp_path / "btc.parquet"
    artifacts_dir = tmp_path / "artifacts"
    make_dataset(dataset_path)

    result, status = run_once(
        dataset_path=dataset_path,
        artifacts_dir=artifacts_dir,
        description="holdout check",
        commit="holdout1",
        run_holdout=True,
    )

    assert status is None
    assert result.holdout is not None
    summary_text = render_summary(result)
    assert "holdout_net_return:" in summary_text
    assert "holdout_btc_return:" in summary_text
    assert "holdout_excess_return_vs_btc:" in summary_text
    assert "holdout_btc_max_drawdown:" in summary_text
    assert "holdout_trade_count:" in summary_text
    summary = json.loads((artifacts_dir / "latest_summary.json").read_text(encoding="utf-8"))
    assert "holdout" in summary
    assert "btc_buy_hold_return" in summary["holdout"]
    assert "excess_return_vs_btc" in summary["holdout"]
    assert "btc_max_drawdown" in summary["holdout"]


def test_current_incumbent_passes_with_new_gate_if_dataset_exists(tmp_path: Path) -> None:
    dataset_path = Path("data/derived/btcusdt_um_1h.parquet")
    if not dataset_path.exists():
        pytest.skip("real dataset not available")

    result, status = run_once(
        dataset_path=dataset_path,
        artifacts_dir=tmp_path / "artifacts",
        results_file=tmp_path / "results.tsv",
        description="incumbent regression",
        commit="regress1",
    )

    assert status == "keep"
    assert result.pass_gates is True
    assert result.active_windows >= EvaluationConfig().min_active_windows
    assert result.worst_window_return >= EvaluationConfig().min_window_net_return
    assert result.net_return > 0.0
