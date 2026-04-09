# Aged Cheddar Cost-Aware Procurement MVP

This project implements a **local-first, Python 3.11 MVP** for weekly aged-cheddar procurement decisions under uncertainty.

It is intentionally framed as a **cost-aware decision system** (not only forecasting):
1. generate synthetic weekly demand/yield/cost data,
2. build strict as-of leakage-safe features,
3. train probabilistic (quantile) forecasts,
4. run rolling-origin backtests (26-week horizon),
5. run Monte Carlo optimization over milk procurement quantities,
6. compare with a naive baseline in business cost terms,
7. generate a guarded markdown memo using only numeric outputs.

## Assumptions

- Planning cadence: weekly.
- Forecast horizon: 26 weeks ahead.
- Procurement decision variable: total milk liters to procure for the planned week.
- Yield maps milk liters to cheese kg using uncertain yield rate.
- Cost function components:
  - milk purchase cost,
  - holding cost for surplus,
  - stockout penalty for unmet demand,
  - spoilage cost on a fraction of surplus.
- Synthetic datasets are realistic but simulated; no external source systems required.

## Project structure

- `src/aged_cheddar_forecasting/` modular source code.
- `data/raw/` generated demand/yield driver datasets.
- `data/external/` generated cost assumptions and pipeline outputs.
- `notebooks/` notebook walkthrough and HTML export target.
- `tests/` basic leakage/cost logic tests.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Run full MVP pipeline

```bash
python -m aged_cheddar_forecasting.pipeline
```

This writes outputs under `data/external/`, including:
- `demand_backtest_predictions.csv`
- `yield_backtest_predictions.csv`
- `decision_cost_curve.csv`
- `procurement_recommendation.csv`
- `procurement_memo.md`

## Run tests

```bash
pytest -q
```

## Notebook

- Open: `notebooks/aged_cheddar_mvp.ipynb`
- Export to HTML:

```bash
jupyter nbconvert --to html notebooks/aged_cheddar_mvp.ipynb --output aged_cheddar_mvp.html
```

## Migration readiness (Phase 2)

The MVP is local-first and keeps business logic in pure Python modules so orchestration/storage can later move to Databricks Jobs or Vertex AI Pipelines with minimal rewrites (mainly I/O/orchestration wrappers).
