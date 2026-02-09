# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Probabilistic Monte Carlo forecasting tool for AGI timelines, based on METR's "Measuring AI Ability to Complete Long Tasks" paper. Runs 100,000 simulations of AI capability growth to produce probability distributions of AGI arrival dates. Has both a Streamlit web UI and Jupyter notebooks for analysis.

## Commands

```bash
# Run the main Streamlit app
streamlit run streamlit_app.py

# Run the acceleration tracking dashboard
streamlit run acceleration_dashboard.py

# Run tests
pytest tests/test_model.py

# Run a single test
pytest tests/test_model.py::test_calculate_doubling_time_simple

# Run Jupyter notebooks
poetry run jupyter notebook

# Install dependencies
pip install -r requirements.txt
# Or with Poetry:
poetry install
```

## Architecture

**Core computation pipeline:**
`model_data.py` (empirical METR benchmark data) → `model_params.py` (parameter distributions) → `model.py` (Monte Carlo simulation) → `plots.py` (visualization) → `streamlit_app.py` (interactive UI)

Key modules:
- **`model.py`** — Core simulation engine. `calculate_doubling_time()` computes days-to-AGI given growth parameters. `run_model(n_samples=100_000)` samples parameter distributions and returns arrays of days/dates. Supports exponential (`acceleration=1`), superexponential (`<1`), and subexponential (`>1`) growth.
- **`model_params.py`** — Default parameter distributions using squigglepy mixture models. Defines `get_start_task_length()`, `get_agi_task_length()`, `get_doubling_time()`, `get_acceleration()`, `get_shift()`.
- **`model_params_ai_rd.py`** / **`chinese_model_params.py`** — Alternative parameter sets for AI R&D and Chinese model scenarios.
- **`model_data.py`** — Dictionary of ~20 AI models (GPT-2 through GPT-5.2 Codex) with launch dates and METR benchmark performance at 50%/80% reliability thresholds.
- **`libs.py`** — Statistical utilities for fitting growth parameters to historical data: `estimate_growth_parameters()`, `bootstrap_growth_parameters()`, `sliding_window_analysis()`.
- **`plots.py`** — Matplotlib visualization: histograms, confidence intervals, growth curves, tornado sensitivity plots.
- **`streamlit_app.py`** — Main interactive UI with sidebar controls, preset models, and tabs for summary/plots/sensitivity/raw data.
- **`acceleration_dashboard.py`** — Separate Streamlit dashboard for analyzing whether AI capability growth is accelerating.

## Key Concepts

- **start_task_length**: Hours the best current AI can reliably complete (sourced from METR benchmarks)
- **agi_task_length**: Hours defining the AGI threshold (e.g., 167h = ~1 month project)
- **doubling_time**: Days for AI task capability to double
- **acceleration**: Modifier on doubling time each period (<1 means progress speeds up)
- **shift**: Days that internal capabilities precede public model release

## Dependencies

Uses `squigglepy` for probabilistic distributions and sampling. Core scientific stack: numpy, pandas, matplotlib, scipy. Streamlit for web UI.

## Python Environment

If you encounter `ModuleNotFoundError` for fundamental packages like `pandas`, `numpy`, `requests`, etc.:
1. **Stop** — Do not attempt to install packages with `pip install`
2. **Alert** — Tell Peter exactly what happened (which module, which command, what you were doing)
3. **Request reboot** — Ask Peter to reboot into the Python sandbox
