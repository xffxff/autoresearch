from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from run_experiment import determine_status, read_best_score, render_summary, run_once


def make_dataset(path: Path, length: int = 24 * 950) -> None:
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
    assert read_best_score(results_path) == pytest.approx(result.score, rel=0, abs=1e-6)
    assert "score:" in render_summary(result, status=status)
    assert determine_status(result, incumbent_score=result.score + 1.0) == "discard"
