# Final Results Table — AutoResearch S&P 500 Direction Prediction

Auto-generated from `results_custom.tsv` — 38 total runs.
**Task**: Predict next-day stock close direction (UP=1 / DOWN=0)
**Data**: 100 S&P 500 stocks, 2020–2024, 41 features
**Split**: 80/20 stratified, random_state=42 (frozen)

---

## Model Comparison

| Model | val_accuracy | val_f1_macro | vs Baseline |
|-------|-------------|-------------|-------------|
| Logistic Regression (baseline) | 0.647648 | 0.645089 | — |
| RandomForest (n=200, depth=8, balanced) | 0.697585 | 0.696775 | ++0.1709 |
| **HistGBT (best: HistGBT depth=6 lr=0.10 iter=1000 balanc)** | **0.724536** | **0.724248** | **++0.0769** |

---

## Best Configuration

| Parameter | Value |
|-----------|-------|
| Model | `HistGradientBoostingClassifier` |
| Description | HistGBT depth=6 lr=0.10 iter=1000 balanced [auto] |
| **val_accuracy** | **0.724536** |
| **val_f1_macro** | **0.724248** |

---

## Improvement Over Baseline

| Metric | Baseline (LR) | Best (HistGBT) | Gain |
|--------|--------------|----------------|------|
| val_accuracy | 0.526723 | 0.724536 | **+0.1978 (+19.8pp)** |
| val_f1_macro | 0.416083 | 0.724248 | **+0.3082 (+30.8pp)** |

---

## Key Finding

Model class was the single most important factor.
Switching from Logistic Regression to HistGradientBoostingClassifier
produced +7.7pp accuracy in one change.
Stock direction is driven by non-linear feature interactions
that LR cannot represent with a linear decision boundary.
