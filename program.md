# BTC strategy autoresearch

This repository is for autonomous research on `BTCUSDT` USD-M perpetual strategies using Binance public data.

## Setup

1. Create a fresh branch named `codex/autoresearch/<YYYYMMDD>-btc-um`.
2. Read the in-scope files:
   - `README.md`
   - `program.md`
   - `backtest.py`
   - `features.py`
   - `strategy.py`
   - `run_experiment.py`
3. Ensure the dataset exists:
   - if raw files are missing, run `uv run download_data.py`
   - if parquet is missing, run `uv run build_dataset.py`
4. Initialize `results.tsv` through the runner:
   - `uv run run_experiment.py --results-file results.tsv --description baseline`
5. Confirm the baseline ran successfully before iterating.

## Scope

You may modify only:

- `features.py`
- `strategy.py`

You must not modify:

- `download_data.py`
- `build_dataset.py`
- `backtest.py`
- `run_experiment.py`
- `pyproject.toml`

## Goal

Maximize the printed research-window `score`, subject to:

- `pass_gates: true`
- higher score is better
- all else equal, prefer simpler logic

The backtest engine is fixed:

- signals are computed on bar close
- trades execute on next bar open
- position range is `[-1, 1]`
- costs are `5 bps` taker fee + `1 bp` slippage per side
- funding is applied from Binance archive `fundingRate`
- score is combined research-window `net_return`
- `net_sharpe` is reported for diagnosis only
- the latest 90 days are reserved as a final holdout window
- the holdout must not be used in the normal keep/discard loop
- hard gates are:
  - `trade_count >= 20`
  - `max_drawdown <= 55%`
  - `annualized_turnover <= 150`
  - `active_windows >= 3`
  - `worst_window_return >= -20%`

## Loop

Repeat forever:

1. Inspect the current branch and the latest `results.tsv`.
2. Make one focused idea in `features.py` or `strategy.py`.
3. Commit the change.
4. Run:

```bash
uv run run_experiment.py --results-file results.tsv --description "<short idea>" > run.log 2>&1
```

5. Extract the summary:

```bash
grep "^score:\|^net_sharpe:\|^max_drawdown:\|^active_windows:\|^worst_window_return:\|^pass_gates:\|^status:" run.log
```

6. If the run crashed, inspect the tail:

```bash
tail -n 80 run.log
```

7. If `status: keep`, keep the commit and continue from it.
8. If `status: discard`, reset to the previous kept commit and try another idea.

## Holdout

Use the final holdout only when you intentionally want a final untouched check for the current branch head.

Run:

```bash
uv run run_experiment.py --run-holdout --description "<short idea>" > run.log 2>&1
```

Do not compare candidates by holdout during routine iteration.

## Results file

`results.tsv` is tab-separated with columns:

```text
commit	scheme	score	net_sharpe	net_return	max_drawdown	trade_count	turnover	active_windows	worst_window_return	pass_gates	status	description
```

Legacy `results.tsv` files are migrated by `run_experiment.py`. Incumbent comparison only happens within the current evaluation `scheme`, so old rows from an earlier evaluator definition are left for audit but are not used as the benchmark.

Crash rows should be appended manually only when the run fails before `run_experiment.py` can write a result.
