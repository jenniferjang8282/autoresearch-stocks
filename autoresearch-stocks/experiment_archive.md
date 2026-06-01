# Experiment Archive — AutoResearch S&P 500 Direction Prediction

Auto-generated from `results_custom.tsv` — 38 total runs.
Pipeline: `prepare_custom.py` (frozen), 41 features, 100 S&P 500 stocks, 2020–2024.
Split: 80/20 stratified, random_state=42. Metric: val_accuracy (higher is better).

**Summary**: 3 baseline | 2 keep | 33 discard
**Best accuracy**: 0.724536 — Run 32: HistGBT depth=6 lr=0.10 iter=1000 balanced [auto]

---

## Full Run Log

| Run | Model | Description | val_acc | val_f1 | Status |
|-----|-------|-------------|---------|--------|--------|
| 01 | LogisticRegression | LR baseline real data + rate + ma5 | 0.526723 | 0.416083 | baseline |
| 02 | LogisticRegression | LR baseline 41 features | 0.522502 | 0.477127 | baseline |
| 03 | LogisticRegression | LR class_weight=balanced | 0.518586 | 0.517958 | discard |
| 04 | HistGBT | HistGBT depth=4 lr=0.05 iter=300 | 0.692754 | 0.688466 | discard |
| 05 | RandomForest | RandomForest n=200 depth=8 balanced | 0.645004 | 0.640141 | discard |
| 06 | HistGBT | HistGBT depth=4 lr=0.05 iter=500 balanced | 0.708263 | 0.708101 | discard |
| 07 | HistGBT | HistGBT iter=500 depth=4 lr=0.05 balanced [wk4 baseline | 0.705975 | 0.705809 | baseline |
| 08 | HistGBT | HistGBT iter=500 depth=2 lr=0.05 balanced | 0.638139 | 0.637871 | discard |
| 09 | HistGBT | HistGBT iter=500 depth=6 lr=0.05 balanced | 0.720620 | 0.720308 | discard |
| 10 | HistGBT | HistGBT iter=500 depth=8 lr=0.05 balanced | 0.720163 | 0.719897 | discard |
| 11 | HistGBT | HistGBT iter=500 depth=4 lr=0.01 balanced | 0.646580 | 0.646271 | discard |
| 12 | HistGBT | HistGBT iter=500 depth=4 lr=0.10 balanced | 0.714925 | 0.714681 | discard |
| 13 | HistGBT | HistGBT iter=500 depth=4 lr=0.05 balanced seed=99 [stab | 0.705111 | 0.704786 | discard |
| 14 | HistGBT | HistGBT depth=6 lr=0.05 iter=500 balanced [auto] | 0.724027 | 0.723671 | keep |
| 15 | HistGBT | HistGBT depth=6 lr=0.10 iter=500 balanced [auto] | 0.723468 | 0.723170 | discard |
| 16 | HistGBT | HistGBT depth=6 lr=0.05 iter=800 balanced [auto] | 0.720875 | 0.720571 | discard |
| 17 | HistGBT | HistGBT depth=6 lr=0.10 iter=800 balanced [auto] | 0.720010 | 0.719660 | discard |
| 18 | HistGBT | HistGBT depth=6 lr=0.05 iter=500 l2=0.5 balanced [auto] | 0.720264 | 0.719966 | discard |
| 19 | HistGBT | HistGBT depth=6 lr=0.05 iter=500 min_samples=30 balance | 0.723163 | 0.722819 | discard |
| 20 | RandomForest | RandomForest n=500 depth=12 balanced [validate] | 0.697585 | 0.696775 | discard |
| 21 | LogisticRegression | ExtraTrees n=500 depth=12 balanced [validate] | 0.643834 | 0.637680 | discard |
| 22 | LogisticRegression | AdaBoost n=200 lr=0.5 [validate] | 0.531909 | 0.378505 | discard |
| 23 | LogisticRegression | BaggingClassifier n=100 depth=8 balanced [validate] | 0.621714 | 0.582281 | discard |
| 24 | LogisticRegression | LinearSVC C=0.1 balanced [validate] | 0.522909 | 0.470420 | discard |
| 25 | LogisticRegression | MLP 128-64 relu balanced [validate] | 0.647648 | 0.645089 | discard |
| 26 | LogisticRegression | RidgeClassifier alpha=1.0 balanced [validate] | 0.519756 | 0.519176 | discard |
| 27 | HistGBT | HistGBT depth=5 lr=0.05 iter=500 balanced [auto] | 0.714518 | 0.714248 | discard |
| 28 | HistGBT | HistGBT depth=6 lr=0.08 iter=600 balanced [auto] | 0.719908 | 0.719590 | discard |
| 29 | RandomForest | RandomForest n=300 depth=10 balanced [auto] | 0.677091 | 0.675065 | discard |
| 30 | HistGBT | HistGBT depth=6 lr=0.05 iter=500 max_bins=128 balanced  | 0.720468 | 0.720133 | discard |
| 31 | HistGBT | HistGBT depth=7 lr=0.05 iter=500 balanced [auto] | 0.719705 | 0.719443 | discard |
| 32 | HistGBT | HistGBT depth=6 lr=0.10 iter=1000 balanced [auto] | 0.724536 | 0.724248 | keep |
| 33 | HistGBT | HistGBT depth=6 lr=0.10 iter=1200 balanced [auto] | 0.722502 | 0.722154 | discard |
| 34 | HistGBT | HistGBT depth=6 lr=0.10 iter=1500 balanced [auto] | 0.722248 | 0.721972 | discard |
| 35 | HistGBT | HistGBT depth=6 lr=0.10 iter=1000 l2=0.1 balanced [auto | 0.722604 | 0.722244 | discard |
| 36 | HistGBT | HistGBT depth=6 lr=0.10 iter=1000 min_samples=20 balanc | 0.723316 | 0.722965 | discard |
| 37 | HistGBT | HistGBT depth=7 lr=0.10 iter=1000 balanced [auto] | 0.719298 | 0.719018 | discard |
| 38 | HistGBT | HistGBT depth=6 lr=0.05 iter=1000 balanced [auto] | 0.721485 | 0.721173 | discard |

---

## Kept Runs

| Run | Description | val_acc | Improvement |
|-----|-------------|---------|-------------|
| 01 | LR baseline real data + rate + ma5 | 0.526723 | baseline |
| 02 | LR baseline 41 features | 0.522502 | baseline |
| 07 | HistGBT iter=500 depth=4 lr=0.05 balanced [wk4 baseline | 0.705975 | baseline |
| 14 | HistGBT depth=6 lr=0.05 iter=500 balanced [auto] | 0.724027 | +0.0181 |
| 32 | HistGBT depth=6 lr=0.10 iter=1000 balanced [auto] | 0.724536 | +0.0005 |

---

## Discarded Runs

| Run | Description | val_acc | vs best at time |
|-----|-------------|---------|-----------------|
| 03 | LR class_weight=balanced | 0.518586 | -0.0081 |
| 04 | HistGBT depth=4 lr=0.05 iter=300 | 0.692754 | +0.1660 |
| 05 | RandomForest n=200 depth=8 balanced | 0.645004 | +0.1183 |
| 06 | HistGBT depth=4 lr=0.05 iter=500 balanced | 0.708263 | +0.1815 |
| 08 | HistGBT iter=500 depth=2 lr=0.05 balanced | 0.638139 | -0.0678 |
| 09 | HistGBT iter=500 depth=6 lr=0.05 balanced | 0.720620 | +0.0146 |
| 10 | HistGBT iter=500 depth=8 lr=0.05 balanced | 0.720163 | +0.0142 |
| 11 | HistGBT iter=500 depth=4 lr=0.01 balanced | 0.646580 | -0.0594 |
| 12 | HistGBT iter=500 depth=4 lr=0.10 balanced | 0.714925 | +0.0090 |
| 13 | HistGBT iter=500 depth=4 lr=0.05 balanced seed=99 [stab | 0.705111 | -0.0009 |
| 15 | HistGBT depth=6 lr=0.10 iter=500 balanced [auto] | 0.723468 | -0.0006 |
| 16 | HistGBT depth=6 lr=0.05 iter=800 balanced [auto] | 0.720875 | -0.0032 |
| 17 | HistGBT depth=6 lr=0.10 iter=800 balanced [auto] | 0.720010 | -0.0040 |
| 18 | HistGBT depth=6 lr=0.05 iter=500 l2=0.5 balanced [auto] | 0.720264 | -0.0038 |
| 19 | HistGBT depth=6 lr=0.05 iter=500 min_samples=30 balance | 0.723163 | -0.0009 |
| 20 | RandomForest n=500 depth=12 balanced [validate] | 0.697585 | -0.0264 |
| 21 | ExtraTrees n=500 depth=12 balanced [validate] | 0.643834 | -0.0802 |
| 22 | AdaBoost n=200 lr=0.5 [validate] | 0.531909 | -0.1921 |
| 23 | BaggingClassifier n=100 depth=8 balanced [validate] | 0.621714 | -0.1023 |
| 24 | LinearSVC C=0.1 balanced [validate] | 0.522909 | -0.2011 |
| 25 | MLP 128-64 relu balanced [validate] | 0.647648 | -0.0764 |
| 26 | RidgeClassifier alpha=1.0 balanced [validate] | 0.519756 | -0.2043 |
| 27 | HistGBT depth=5 lr=0.05 iter=500 balanced [auto] | 0.714518 | -0.0095 |
| 28 | HistGBT depth=6 lr=0.08 iter=600 balanced [auto] | 0.719908 | -0.0041 |
| 29 | RandomForest n=300 depth=10 balanced [auto] | 0.677091 | -0.0469 |
| 30 | HistGBT depth=6 lr=0.05 iter=500 max_bins=128 balanced  | 0.720468 | -0.0036 |
| 31 | HistGBT depth=7 lr=0.05 iter=500 balanced [auto] | 0.719705 | -0.0043 |
| 33 | HistGBT depth=6 lr=0.10 iter=1200 balanced [auto] | 0.722502 | -0.0020 |
| 34 | HistGBT depth=6 lr=0.10 iter=1500 balanced [auto] | 0.722248 | -0.0023 |
| 35 | HistGBT depth=6 lr=0.10 iter=1000 l2=0.1 balanced [auto | 0.722604 | -0.0019 |
| 36 | HistGBT depth=6 lr=0.10 iter=1000 min_samples=20 balanc | 0.723316 | -0.0012 |
| 37 | HistGBT depth=7 lr=0.10 iter=1000 balanced [auto] | 0.719298 | -0.0052 |
| 38 | HistGBT depth=6 lr=0.05 iter=1000 balanced [auto] | 0.721485 | -0.0031 |
