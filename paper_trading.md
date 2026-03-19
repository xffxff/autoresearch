# Hyperliquid Paper Trading

This repository keeps offline research and venue validation separate.

- `run_experiment.py` remains the offline research evaluator.
- `run_paper.py` is a separate Hyperliquid mainnet paper runner.
- Only approved strategy snapshots under `deployments/paper/` can be used for paper trading.
- Paper execution is pure taker with fixed non-compounding capital of `10000 USDC`.
- Implementation code is organized under the `paper_trading/` package; `run_paper.py` and `approve_paper.py` are thin wrappers.

## Approval flow

Create a paper snapshot from the current research surface:

```bash
uv run approve_paper.py --approved-by <name> --description "paper candidate"
```

This copies `features.py` and `strategy.py` into `deployments/paper/versions/<version_id>/`, writes a manifest with hashes, and updates `deployments/paper/current.json`.

## Running paper

Run one paper-trading tick:

```bash
uv run run_paper.py
```

Run continuously:

```bash
uv run run_paper.py --loop --poll-interval-seconds 30
```

The runner:

- loads the approved paper snapshot
- warms up from recent Hyperliquid candles
- rebuilds a standard market frame
- computes the latest target position
- simulates pure taker execution using current top-of-book quotes
- writes paper artifacts to `paper_artifacts/`

Create `paper_artifacts/HALT` to stop future paper actions without changing the approved deployment.

## Artifacts

`paper_artifacts/` contains:

- `latest_status.json`
- `runtime_state.json`
- `decisions.tsv`
- `fills.tsv`
- `equity.tsv`
- `risk_events.tsv`
- `daily/<YYYY-MM-DD>.json`

These artifacts are intentionally separate from offline research outputs like `results.tsv` and `artifacts/latest_summary.json`.

## Deployment layout

Approved paper snapshots live under `deployments/paper/`:

```text
deployments/paper/current.json
deployments/paper/versions/<version_id>/features.py
deployments/paper/versions/<version_id>/strategy.py
deployments/paper/versions/<version_id>/manifest.json
```
