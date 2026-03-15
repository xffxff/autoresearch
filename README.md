# BTC Strategy Autoresearch

This repository adapts the core idea from [karpathy/autoresearch](https://github.com/karpathy/autoresearch) to offline BTC strategy research on Binance USD-M futures.

The project keeps a hard boundary between:

- fixed infrastructure: data download, dataset build, backtest, experiment runner
- agent-editable research surface: `features.py`, `strategy.py`
- human-editable instructions: `program.md`

## Workflow

1. Download Binance public BTCUSDT USD-M data:

```bash
uv run download_data.py
```

2. Build the normalized 1h dataset:

```bash
uv run build_dataset.py
```

3. Run the current strategy:

```bash
uv run run_experiment.py --results-file results.tsv --description baseline
```

The default evaluation uses:

- `BTCUSDT` USD-M perpetual
- `1h` bars
- 6 walk-forward windows
- 365 day calibration + 90 day validation per window
- score = combined validation `net_return`
- survival gates:
  - `trade_count >= 20`
  - `max_drawdown <= 55%`
  - `annualized_turnover <= 150`
  - `active_windows >= 3`
  - `worst_window_return >= -20%`

Implementation note: Binance public `fundingRate` is available from monthly archives, not daily archives. When a trailing funding archive has not been published yet, the dataset builder zero-fills those missing rows.

## Project layout

- `download_data.py`: downloads monthly full files plus daily incremental files from Binance public archives and verifies checksums
- `build_dataset.py`: normalizes raw zip files into a single aligned parquet dataset
- `backtest.py`: fixed market bundle types, walk-forward evaluator, cost model, and risk gates
- `features.py`: agent-editable feature builder
- `strategy.py`: agent-editable position generator
- `run_experiment.py`: fixed experiment entrypoint that prints a summary and can append to `results.tsv`
- `program.md`: instructions for a coding agent to run the loop

## Tests

```bash
uv run pytest
```
