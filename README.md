[README.md](https://github.com/user-attachments/files/28469485/README.md)

# AutoResearch: Top-100 S&P 500 Stock Close Direction Prediction

Predict whether each stock's **closing price will go UP or DOWN** relative to the previous day.
**Best result**: 72.45% validation accuracy (HistGBT depth=6, lr=0.10, iter=1000).
**Data**: real OHLCV prices via yfinance + FEDFUNDS interest rate via FRED (2020–2024).

---

## The Claim

**HistGradientBoostingClassifier is the best model class for this task.**

Across 38 controlled experiments and 9 model families, HistGBT outperformed every alternative
tested — including Random Forest, MLP, SVM, AdaBoost, and Logistic Regression.
The +19.8pp gap over Logistic Regression came from one model class change, not from
feature engineering or hyperparameter tuning.

---

## Problem

Given 41 technical and market features for a stock on a given trading day, predict whether
the stock's closing price will be higher or lower the next day.

- **Input:** 41 features per stock-day (returns, volatility, moving averages, volume, momentum, interest rate, SPY/VIX market context)
- **Target:** Binary label — `1` = close goes UP, `0` = close goes DOWN
- **Dataset:** 100 S&P 500 stocks x ~1,000 trading days (2020–2024) — ~98,000 labeled samples
- **Split:** 80/20 stratified, random_state=42 (frozen across all experiments)
- **Primary metric:** Validation Accuracy
- **Secondary metric:** F1-Macro

---

## Project Structure

```
autoresearch-stocks/
│
├── model.py                  # EDITABLE — the only file the agent modifies
│
├── prepare_custom.py         # FROZEN — real-data pipeline (yfinance + FRED, 41 features)
├── run_custom.py             # FROZEN — experiment runner, logs to results_custom.tsv
├── fred_rates.csv            # FEDFUNDS interest rate data (2020–2024)
├── program.md                # Agent instructions — Init / Loop / Finalize structure
│
├── autonomous_loop.py        # Runs 6 experiments automatically
│
├── results_custom.tsv        # Full experiment log (38 runs)
├── experiment_archive.md     # Annotated log of all runs with decisions
├── final_results_table.md    # Clean model comparison table
├── performance.png           # Accuracy + F1 over all runs (auto-generated)
└── model_comparison.png      # Best accuracy per model class (auto-generated)
```

**Key rule**: the agent may only modify `model.py`. Everything else is frozen.

---

## Setup

### Install Python dependencies

Requires **Python 3.10+**.

```bash
pip install scikit-learn matplotlib numpy pandas yfinance
```

| Package | Purpose |
|---------|---------|
| scikit-learn | ML models, pipelines, evaluation |
| matplotlib | Performance plotting |
| numpy / pandas | Data manipulation |
| yfinance | Real OHLCV price data for 100 S&P 500 stocks + SPY + VIX |

### FRED interest rate data

`fred_rates.csv` is already included (FEDFUNDS 2020–2024).
To update: download a new CSV from [fred.stlouisfed.org](https://fred.stlouisfed.org) and replace the file.

### Verify setup

```bash
python run_custom.py "test run" --discard
# Expected:
#   Data: 78,657 train | 19,665 val | 41 features
#   val_accuracy: 0.724xxx
#   Result logged to results_custom.tsv (status=discard)
```

---

## How to Run

### Run the autonomous loop (6 experiments at a time)

```bash
python autonomous_loop.py
```

Follows a strict 3-phase structure:

**INIT** — reads `program.md` for rules, finds current best from `results_custom.tsv`

**LOOP x6** — for each candidate config:
1. Edits `model.py` with one change
2. Runs `python run_custom.py "<description>"`
3. val_accuracy improved → KEEP / not improved → REVERT `model.py`

**FINALIZE** — regenerates all charts and docs from `results_custom.tsv`

### Run a single experiment manually

```bash
python run_custom.py "description of change"
# flags: --baseline (first run only), --discard (no improvement)
```

---

## Results

### Best configuration

| Parameter | Value |
|-----------|-------|
| Model | `HistGradientBoostingClassifier` |
| `max_depth` | 6 |
| `learning_rate` | 0.10 |
| `max_iter` | 1000 |
| `class_weight` | `"balanced"` |
| `random_state` | 42 |
| **val_accuracy** | **0.7245** |
| **val_f1_macro** | **0.7242** |

### Model class comparison

| Model | Best val_accuracy | vs. LR baseline |
|-------|------------------|----------------|
| Logistic Regression | 0.5267 | — |
| Ridge Classifier | 0.5198 | -0.7pp |
| AdaBoost | 0.5319 | +0.5pp |
| LinearSVC | 0.5229 | -0.4pp |
| Bagging (DT) | 0.6217 | +9.5pp |
| MLP (128-64) | 0.6476 | +12.1pp |
| Extra Trees | 0.6438 | +17.1pp |
| Random Forest | 0.6976 | +17.1pp |
| **HistGBT** | **0.7245** | **+19.8pp** |

---

## Features (41 total)

| Group | Features |
|-------|---------|
| Returns (6) | `ret_1`, `ret_2`, `ret_3`, `ret_5`, `ret_10`, `ret_20` |
| Volatility (3) | `vol_5`, `vol_10`, `vol_20` |
| Moving averages (4) | `ma_5_ratio`, `ma_10_ratio`, `ma_20_ratio`, `ma_5_vs_20` |
| OHLCV intraday (6) | `intraday_ret`, `overnight_gap`, `hl_range`, `close_to_high`, `close_to_low`, `body_ratio` |
| Volume (5) | `log_vol`, `vol_ratio`, `vol_ratio_20`, `vol_change_1d`, `vol_volatility` |
| Momentum (4) | `price_pos`, `rsi_like`, `macd`, `rate` |
| Direction & streak (3) | `up_yesterday`, `up_days_10`, `streak` |
| SPY & VIX (10) | `spy_ret_1/5/20`, `spy_vol_10`, `rel_ret_1/5/20`, `rel_vol_10`, `vix_level`, `vix_change` |

---

## Files Reference

| File | Description |
|------|-------------|
| `model.py` | Current best model config — the only editable file |
| `prepare_custom.py` | Frozen pipeline: downloads prices, engineers 41 features, splits data |
| `run_custom.py` | Frozen runner: trains model, evaluates, appends to results_custom.tsv |
| `fred_rates.csv` | FEDFUNDS monthly interest rate data, forward-filled to daily |
| `program.md` | Agent rules: Init/Loop/Finalize structure, keep/discard/crash rule, search strategy |
| `autonomous_loop.py` | Automated loop: runs 6 experiments, updates all outputs at the end |
| `results_custom.tsv` | Append-only log of every experiment run |
| `experiment_archive.md` | Human-readable log with decisions and reasons |
| `final_results_table.md` | Clean model comparison table, auto-generated |
| `performance.png` | Accuracy and F1 across all runs, auto-generated |
| `model_comparison.png` | Best accuracy per model class, auto-generated |
