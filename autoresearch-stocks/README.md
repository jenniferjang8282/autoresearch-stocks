# AutoResearch: Top-100 S&P 500 Stock Close Direction Prediction

Predict whether each stock's **closing price will go UP or DOWN** relative to the previous day.
**Metric**: validation accuracy (higher is better).
**Data**: real OHLCV prices via yfinance + FEDFUNDS interest rate via FRED (2020–2024).

---

## Problem

Given 41 technical and market features for a stock on a given trading day, predict whether
the stock's closing price will be higher or lower the next day.

- **Input:** 41 features per stock-day (returns, volatility, moving averages, volume, momentum, interest rate, SPY/VIX market context)
- **Target:** Binary label — `1` = close goes UP, `0` = close goes DOWN
- **Dataset:** 100 S&P 500 stocks × ~1,000 trading days (2020–2024) → ~98,000 labeled samples (80/20 split)
- **Primary metric:** Validation Accuracy (higher is better)
- **Secondary metric:** F1-Macro

---

## Project Structure

```
autoresearch-stocks/
│
├── model.py                  # EDITABLE — agent modifies only this file
│
├── prepare_custom.py         # FROZEN — real-data pipeline (yfinance + FRED, 41 features)
├── run_custom.py             # FROZEN — experiment runner, logs to results_custom.tsv
├── fred_rates.csv            # FEDFUNDS interest rate data (2020–2024)
├── program.md                # Agent instructions (includes autonomous block docs)
│
├── autonomous_loop.py        # Autonomous block — runs 6 experiments automatically
├── autonomous_report.md      # Generated report from the last autonomous run
│
├── results_custom.tsv        # Experiment log (primary)
│
├── analyze.py                # Generates experiment matrix + plot + memo
├── experiment_matrix.tsv     # Full results with FPR, FNR, overfit gap, log-loss
├── performance.png           # Performance plot (auto-generated)
├── failure_analysis_memo.md  # Error taxonomy and findings
│
├── prepare.py                # Legacy — original synthetic data pipeline
├── run.py                    # Legacy — synthetic experiment runner
├── results.tsv               # Legacy — synthetic experiment log
└── demo.py                   # Legacy — 8-iteration synthetic demo
```

**Key rule**: the agent may only modify `model.py`. Everything else is frozen.

---

## Setup

### 1. Install an AI coding agent (CLI)

#### Option A: Claude Code CLI (recommended)

```bash
# macOS / Linux
curl -fsSL https://claude.ai/install.sh | bash

# macOS via Homebrew
brew install --cask claude-code

# Windows PowerShell
irm https://claude.ai/install.ps1 | iex
```

Then launch:

```bash
cd autoresearch-stocks
claude
```

First launch opens a browser for login — no API key needed.
Works with any Claude subscription (Pro, Max, or Team).

#### Option B: OpenAI Codex CLI

```bash
npm install -g @openai/codex
cd autoresearch-stocks
codex
```

### 2. Install Python dependencies

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

### 3. FRED interest rate data

`fred_rates.csv` is already included in this folder (FEDFUNDS 2020–2024).
To update it: download a new CSV from [fred.stlouisfed.org](https://fred.stlouisfed.org)
(search FEDFUNDS or DGS10) and replace the file.

### 4. Verify setup

```bash
python run_custom.py "test run" --discard
# Expected output:
#   Data: 78,657 train | 19,665 val | 41 features
#   val_accuracy: 0.52xxxx
#   val_f1_macro: 0.4xxxxx
#   Result logged to results_custom.tsv (status=discard)
```

---

## How to Run the Agent Loop

### Quick start (copy-paste into your agent)

```
Read program.md for instructions, then read model.py.
Run `python run_custom.py "baseline" --baseline` to establish the baseline accuracy.
Then enter the AutoResearch loop:

1. Propose one modification to model.py (different classifier,
   hyperparameter change, class-weight adjustment, etc.).
2. Edit model.py with your change.
3. Run: python run_custom.py "<short description>"
4. Compare val_accuracy to the current best.
   - If improved: KEEP the change, note the new best.
   - If worse: REVERT model.py to the previous version.
5. Repeat from step 1. Try at least 6 different ideas.

After all iterations, run `python analyze.py` to generate performance.png.
Print a summary table of all experiments and which were kept vs discarded.
```

### More specific prompt (if you want to control the search)

```
You are an AutoResearch agent. Read program.md for rules.

Your job: maximize val_accuracy on S&P 500 close-direction prediction
by modifying model.py.

Constraints:
- model.py must define build_model() returning an sklearn-compatible estimator
- Do NOT modify prepare_custom.py, run_custom.py, or any frozen file
- Each experiment must finish in < 180 seconds

Search strategy:
1. Start with the current baseline (LogisticRegression + StandardScaler)
2. Try class-weight balancing (class_weight="balanced") to improve F1
3. Try other scalers (RobustScaler, QuantileTransformer)
4. Try tree ensembles (RandomForest, HistGradientBoosting)
5. Tune the best model so far (learning rate, depth, iterations)
6. Try an ensemble (VotingClassifier combining best models)

For each experiment:
- Run: python run_custom.py "<description>"
- If val_accuracy improved → keep
- If val_accuracy worsened → revert model.py
- Log your reasoning for each decision

After finishing, run: python analyze.py
```

---

## Example Agent Loop (actually executed)

Below is a real session using the real-data pipeline (`run_custom.py`).
The agent modified `model.py` across 9 experiments.

### Iteration 0 — Baseline

```python
# model.py
Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(C=1.0, max_iter=1000))])
```

```
$ python run_custom.py "LR baseline real data + rate + ma5" --baseline
val_accuracy: 0.526723   val_f1_macro: 0.416083
```

**BASELINE established**: accuracy = 0.5267.
F1 is low (0.416) — model is biased toward predicting UP since 52% of days are up days.

---

### Controlled experiment set — Factor A: Scaler

Held model fixed (LR C=1.0), varied preprocessing.

| Scaler | val_accuracy | val_f1 | Decision |
|--------|-------------|--------|----------|
| StandardScaler (baseline) | 0.5031 | 0.4896 | baseline |
| None | 0.4983 | 0.4814 | discard |
| RobustScaler | 0.5035 | 0.4900 | discard |
| QuantileTransformer | 0.4990 | 0.4829 | discard |

**Finding**: scaler choice has minimal effect on accuracy but removing it hurts F1.

---

### Controlled experiment set — Factor B: Regularisation

Held scaler fixed (StandardScaler), varied C in LogisticRegression.

| C | val_accuracy | val_f1 | Decision |
|---|-------------|--------|----------|
| 0.01 | 0.4983 | 0.4812 | discard |
| 0.1 | 0.5025 | 0.4885 | discard |
| 1.0 (baseline) | 0.5031 | 0.4896 | baseline |
| 10.0 | 0.5030 | 0.4895 | discard |

**Finding**: LR accuracy is flat across all C values — the data is not linearly separable.

---

### Controlled experiment set — Factor C: Architecture

Held scaler fixed (StandardScaler), varied model class.

| Model | val_accuracy | val_f1 | Decision |
|-------|-------------|--------|----------|
| LogisticRegression (baseline) | 0.5031 | 0.4896 | baseline |
| RandomForest (n=200) | 0.4984 | 0.4984 | discard |
| HistGBT (max_iter=300) | 0.4982 | 0.4737 | discard |

**Finding**: LR remains competitive. RandomForest has the best F1 but overfits (train acc = 100%).

---

### Iteration 12 — Real data, 41 features

Expanded feature set from 15 → 41 (added SPY/VIX market features, volume features, streak, intraday features).

```
$ python run_custom.py "LR baseline 41 features" --baseline
val_accuracy: 0.522502   val_f1_macro: 0.477127
```

**Result**: F1 improved from 0.416 → 0.477 with more features. Accuracy stable.

---

### Summary

| # | Model | val_accuracy | val_f1 | Decision |
|---|-------|-------------|--------|----------|
| 0 | LR + StandardScaler (15 features) | 0.5267 | 0.4161 | baseline |
| 1–3 | LR, scaler variants (none / Robust / Quantile) | 0.498–0.504 | 0.481–0.490 | discard |
| 4–6 | LR, C variants (0.01 / 0.1 / 10.0) | 0.498–0.503 | 0.481–0.490 | discard |
| 7–8 | RF(n=200) / HistGBT(iter=300) | 0.498–0.499 | 0.474–0.498 | discard |
| 9 | LR + StandardScaler (41 features) | 0.5225 | **0.4771** | baseline |

**Best accuracy to date: 0.5267 (LR + StandardScaler, 15 features)**

---

## Plotting Results

After running experiments:

```bash
python analyze.py
# Generates:
#   performance.png         — accuracy and F1 over all experiments
#   experiment_matrix.tsv   — full results with FPR, FNR, overfit gap, log-loss
#   failure_analysis_memo.md — error taxonomy and factor findings
```

The performance chart shows:
- **Top panel**: validation accuracy over iterations
- **Bottom panel**: validation F1-Macro over iterations
- **Colour coding**: blue = baseline, green = keep, red = discard
- **Green line**: best-so-far envelope

---

## Features (41 total — real-data pipeline)

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

## Autonomous Block

Run 6 experiments automatically with a single command:

```bash
python autonomous_loop.py
```

The block:
1. Reads `results_custom.tsv` to find the current best accuracy
2. Picks the next 6 untried configs from a predefined search space (already-tried ones are skipped)
3. For each config: writes it to `model.py`, runs the evaluation, then **KEEP** / **DISCARD** / **CRASH**:
   - **KEEP** — accuracy improved → saves the new config as the checkpoint
   - **DISCARD** — no improvement → reverts `model.py` to the previous best
   - **CRASH** — timeout or error → reverts `model.py`, records the error message
4. Writes `autonomous_report.md` with the full outcome table, decision log, and findings

To extend the search space, add `(description, model_py_body)` tuples to `SEARCH_SPACE` at the top of `autonomous_loop.py`.

---

## Ideas to Explore

- `class_weight="balanced"` — addresses the 52% UP-day bias and improves F1
- `HistGradientBoostingClassifier` with tuned depth and learning rate
- `VotingClassifier` combining LR + HistGBT
- `CalibratedClassifierCV` for better probability estimates
- Feature selection — remove low-importance features to reduce noise
- Threshold tuning — shift decision boundary away from 0.5 to optimise F1
