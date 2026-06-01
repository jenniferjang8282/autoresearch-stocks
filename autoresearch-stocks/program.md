# AutoResearch — Program Instructions

**Task**: Predict whether each S&P 500 stock's closing price will go UP (1) or DOWN (0) the next day.
**Metric**: `val_accuracy` (higher is better). Secondary: `val_f1_macro`.
**Editable file**: `model.py` only. All other files are frozen.

---

## INIT (once)

1. Read this file (`program.md`) — understand the rules and search strategy
2. Run the baseline to establish the current best:

```bash
python run_custom.py "baseline" --baseline
```

If `results_custom.tsv` already has runs, read the best `val_accuracy` from the `keep` or `baseline` rows instead of re-running.

**Current best to beat**: `0.724536` — HistGBT depth=6 lr=0.10 iter=1000 balanced

---

## LOOP (≥ 6 iterations)

Repeat for each experiment:

### Step 1 — Edit model.py
Make exactly **one change** to `build_model()`. Choose from the search strategy below.

### Step 2 — Run evaluation
```bash
python run_custom.py "<short description of change>"
```

### Step 3 — Compare val_accuracy vs current best
- **val_accuracy improved** → **KEEP**: this is the new best. Move to next iteration.
- **val_accuracy did not improve** → **REVERT**: restore `model.py` to the previous version.

```bash
# To revert:
git checkout model.py
```

Repeat from Step 1.

---

## FINALIZE (once)

After all iterations, regenerate all outputs:

```bash
python generate_charts.py
```

This reads `results_custom.tsv` and updates:
- `performance.png` — accuracy + F1 over all runs
- `model_comparison.png` — best accuracy per model class
- `experiment_archive.md` — full run log
- `final_results_table.md` — clean comparison table

---

## Rules

| Rule | Detail |
|------|--------|
| Only edit `model.py` | All other files are frozen |
| One change per experiment | Change exactly one parameter at a time |
| Always revert on no improvement | Use `git checkout model.py` |
| Never modify `prepare_custom.py` | This would break comparability across runs |
| Never modify `run_custom.py` | Frozen evaluator |
| Timeout limit | If a run takes > 240 seconds, revert and try something else |

---

## Search Strategy

The current best model is `HistGradientBoostingClassifier` (depth=6, lr=0.10, iter=1000).
All future experiments should stay within the HistGBT family.

**Priority order** (try these first — none have been attempted yet):

| Priority | What to try | Why |
|----------|------------|-----|
| 1 | `max_iter=1200` with lr=0.10, depth=6 | More iterations may still help |
| 2 | `max_iter=1500` with lr=0.10, depth=6 | Push further |
| 3 | `l2_regularization=0.1` on best config | Reduce overfitting |
| 4 | `min_samples_leaf=20` on best config | Alternative regularisation |
| 5 | `max_depth=7` with lr=0.10, iter=1000 | Slightly deeper |
| 6 | `max_iter=1000` with lr=0.05, depth=6 | Compare lr=0.05 at same iterations |

**Do not try:**
- Logistic Regression — confirmed bottleneck at 52.7%
- RandomForest — confirmed below HistGBT (69.8% best)
- `max_depth > 8` — diminishing returns shown at depth=8
- `learning_rate=0.01` — signal failure at 500 iterations confirmed
- Any non-sklearn model — outside scope

---

## Data

| Property | Value |
|----------|-------|
| Source | yfinance (100 S&P 500 stocks + SPY + VIX, 2020–2024) + FRED interest rates |
| Features | 41 (returns, volatility, moving averages, volume, momentum, SPY/VIX, FEDFUNDS) |
| Train samples | 78,657 |
| Val samples | 19,665 |
| Split | 80/20, stratified, random_state=42 (frozen) |
| Class balance | ~52% UP, ~48% DOWN |

---

## Logging

Every run is automatically appended to `results_custom.tsv` by `run_custom.py`.

| Column | Description |
|--------|-------------|
| `experiment` | Git hash or `no-git` |
| `val_acc` | Validation accuracy (6 decimal places) |
| `val_f1` | Validation F1-macro (6 decimal places) |
| `status` | `baseline`, `keep`, or `discard` |
| `description` | The description string passed on the command line |

Use `--baseline` only for the very first run. Use `--discard` for runs that do not improve. Default (no flag) = `keep`.
