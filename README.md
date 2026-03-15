# btc-autoresearch

This repository adapts the core idea from [karpathy/autoresearch](https://github.com/karpathy/autoresearch) to offline BTC strategy research on Binance USD-M futures.

The idea: give an AI agent a small but real strategy research setup and let it iterate autonomously. It edits `features.py` and `strategy.py`, runs a fixed walk-forward backtest, checks whether the result improved, keeps or discards the experiment, and repeats. The human mainly edits `program.md` and reviews the resulting research log.

You wake up to a `results.tsv`, a `run.log`, a set of artifacts, and hopefully a better BTC strategy.

## How it works

The repo is intentionally small and keeps a hard boundary between fixed infrastructure and the agent-editable research surface:

- `download_data.py` downloads Binance public BTCUSDT USD-M data and verifies checksums. Not modified.
- `build_dataset.py` builds the normalized 1h parquet dataset. Not modified.
- `backtest.py` defines the evaluator, execution rules, and gates. Not modified.
- `run_experiment.py` runs the strategy, prints the summary, and appends to `results.tsv`. Not modified.
- `features.py` is edited by the agent.
- `strategy.py` is edited by the agent.
- `program.md` is edited by the human and acts as the lightweight research operating manual.

By design, the data and evaluator are fixed. The score is the combined research-window `net_return`, and a candidate is only kept if it also passes the survival gates:

- `trade_count >= 20`
- `max_drawdown <= 55%`
- `annualized_turnover <= 150`
- `active_windows >= 3`
- `worst_window_return >= -20%`

The evaluator also reserves a final 90-day holdout window after the research windows. That holdout is not shown during the normal keep/discard loop unless you explicitly ask for it.

The default research target is:

- `BTCUSDT` USD-M perpetual
- `1h` bars
- 8 research walk-forward windows
- 1 final 90 day holdout window
- 365 day calibration + 90 day validation per research window
- next-bar-open execution
- `5 bps` taker fee + `1 bp` slippage per side
- Binance archived funding applied as holding cost/carry

## Quick start

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/), internet access for Binance public archives.

```bash
# 1. Install dependencies
uv sync --dev

# 2. Download BTCUSDT USD-M raw archives
uv run download_data.py

# 3. Build the normalized 1h dataset
uv run build_dataset.py

# 4. Run one experiment
uv run run_experiment.py --results-file results.tsv --description baseline

# 5. When you want a final untouched check, reveal the holdout once
uv run run_experiment.py --run-holdout --description final-check
```

If those commands work, the setup is ready for autonomous research.

## Running the agent

Open your coding agent in this repository and point it at `program.md`. A typical prompt is:

```text
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The agent should read `program.md`, inspect the latest `results.tsv`, make one focused idea in `features.py` or `strategy.py`, run the fixed research experiment command, and keep or discard based on the reported `status`. It should only reveal the final holdout window intentionally, not every iteration.

## Project structure

```text
download_data.py   data download + checksum verification (do not modify)
build_dataset.py   raw archive normalization into 1h parquet (do not modify)
backtest.py        fixed evaluator, execution model, and gates (do not modify)
run_experiment.py  fixed experiment runner and results logger (do not modify)
features.py        feature engineering surface (agent modifies this)
strategy.py        position logic surface (agent modifies this)
program.md         human instructions for the agent
pyproject.toml     dependencies
results.tsv        research experiment log
artifacts/         latest summary JSON + research-window breakdown (+ optional holdout)
```

## Design choices

- **Fixed evaluator.** Agents are not allowed to game the benchmark by changing the backtest engine, data source, or scoring pipeline.
- **Small editable surface.** Research happens in `features.py` and `strategy.py`, which keeps diffs readable and the search space manageable.
- **Walk-forward validation.** Every experiment is scored on sequential out-of-sample windows instead of a single in-sample period.
- **Reserved holdout.** The latest 90 days are held back from the research score and can be revealed separately for a final sanity check.
- **Execution realism.** Signals are computed on bar close and executed on the next bar open, with costs and funding applied.
- **Keep/discard loop.** Each run appends to `results.tsv`, which makes the research trajectory auditable and easy to resume.

## Data notes

Binance public `fundingRate` is available from monthly archives, not daily archives. When a trailing monthly funding archive has not been published yet, the dataset builder zero-fills those missing rows.

Raw data comes from:

- [binance-public-data](https://github.com/binance/binance-public-data)
- [data.binance.vision](https://data.binance.vision/)

## Tests

```bash
uv run pytest
```
