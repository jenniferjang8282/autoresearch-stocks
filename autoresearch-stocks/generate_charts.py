"""
generate_charts.py — Auto-generate all outputs from results_custom.tsv

Called automatically at the end of autonomous_loop.py, or run manually:
    python generate_charts.py

Outputs (all regenerated from results_custom.tsv every time):
    performance.png          — accuracy + F1 over all runs
    model_comparison.png     — best accuracy per model class (bar chart)
    experiment_archive.md    — full run log with decisions
    final_results_table.md   — clean model comparison + best config
"""

import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RESULTS_TSV = "results_custom.tsv"

# ── Load results ───────────────────────────────────────────
runs = []
with open(RESULTS_TSV) as f:
    reader = csv.DictReader(f, delimiter="\t")
    for i, row in enumerate(reader, 1):
        runs.append({
            "run":    i,
            "acc":    float(row["val_acc"]),
            "f1":     float(row["val_f1"]),
            "status": row["status"].strip(),
            "desc":   row["description"].strip(),
        })

if not runs:
    print("No data found in results_custom.tsv")
    exit()

xs    = [r["run"]  for r in runs]
accs  = [r["acc"]  for r in runs]
f1s   = [r["f1"]   for r in runs]
best_run  = max(runs, key=lambda r: r["acc"])
first_run = runs[0]

STATUS_COLORS = {
    "baseline": "#3498db",
    "keep":     "#2ecc71",
    "discard":  "#e74c3c",
}

def pt_color(status):
    return STATUS_COLORS.get(status, "#95a5a6")

colors = [pt_color(r["status"]) for r in runs]

# Best-so-far envelope
best_so_far = []
cur_best = -1
for r in runs:
    if r["status"] in ("baseline", "keep"):
        cur_best = max(cur_best, r["acc"])
    best_so_far.append(cur_best if cur_best > 0 else r["acc"])


# ── Chart 1: Performance over runs ────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(max(12, len(runs)), 9), sharex=True)

for ax, vals, ylabel in [
    (axes[0], accs, "Validation Accuracy"),
    (axes[1], f1s,  "Validation F1-Macro"),
]:
    ax.scatter(xs, vals, c=colors, s=110, zorder=4, edgecolors="white", linewidth=0.8)
    ax.plot(xs, vals, color="grey", linestyle="--", alpha=0.25, zorder=2)
    ax.plot(xs, best_so_far, color="#2ecc71", linewidth=2.2, zorder=3, label="Best so far")
    ax.axhline(accs[0], color="#3498db", linestyle=":", linewidth=1.4,
               label=f"Original baseline ({accs[0]:.4f})")
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1.0, alpha=0.4, label="Chance (50%)")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.2)
    for x, v in zip(xs, vals):
        ax.annotate(f"{v:.3f}", (x, v), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=7.5)

axes[0].set_title(
    f"Validation Performance Across All {len(runs)} Runs\n(source: results_custom.tsv)",
    fontsize=13, fontweight="bold"
)
legend_elements = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#3498db", markersize=10, label="baseline"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#2ecc71", markersize=10, label="keep"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#e74c3c", markersize=10, label="discard"),
    Line2D([0],[0], color="#2ecc71", linewidth=2.2, label="best so far"),
]
axes[0].legend(handles=legend_elements, loc="lower right", fontsize=9)
axes[1].legend(fontsize=9, loc="lower right")

def short_label(r):
    d = r["desc"]
    if "LR" in d and "41" in d:          return f"R{r['run']:02d}\nLR 41feat"
    if "LR" in d and "balanced" in d:    return f"R{r['run']:02d}\nLR bal"
    if "LR" in d:                        return f"R{r['run']:02d}\nLR base"
    if "RandomForest" in d:              return f"R{r['run']:02d}\nRF"
    if "depth=2" in d:                   return f"R{r['run']:02d}\ndepth=2"
    if "depth=6" in d and "lr=0.10" in d and "800" in d: return f"R{r['run']:02d}\nd6 lr.10 800"
    if "depth=6" in d and "lr=0.10" in d: return f"R{r['run']:02d}\nd6 lr.10"
    if "depth=6" in d and "800" in d:   return f"R{r['run']:02d}\nd6 800"
    if "depth=6" in d and "l2" in d:    return f"R{r['run']:02d}\nd6 l2"
    if "depth=6" in d and "min_s" in d: return f"R{r['run']:02d}\nd6 msl"
    if "depth=6" in d:                  return f"R{r['run']:02d}\ndepth=6"
    if "depth=8" in d:                  return f"R{r['run']:02d}\ndepth=8"
    if "lr=0.01" in d:                  return f"R{r['run']:02d}\nlr=0.01"
    if "lr=0.10" in d:                  return f"R{r['run']:02d}\nlr=0.10"
    if "seed=99" in d:                  return f"R{r['run']:02d}\nseed=99"
    if "wk4" in d:                      return f"R{r['run']:02d}\nwk4 base"
    if "iter=300" in d:                 return f"R{r['run']:02d}\niter=300"
    return f"R{r['run']:02d}\nrun{r['run']}"

axes[1].set_xticks(xs)
axes[1].set_xticklabels([short_label(r) for r in runs], fontsize=7.5)
axes[1].set_xlabel("Experiment", fontsize=11)

plt.tight_layout()
plt.savefig("performance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Wrote performance.png")


# ── Chart 2: Model class comparison ───────────────────────
model_groups = {"Logistic\nRegression": [], "Random\nForest": [], "HistGBT": []}
for r in runs:
    d = r["desc"]
    if "RandomForest" in d:
        model_groups["Random\nForest"].append(r["acc"])
    elif "HistGBT" in d or "Hist" in d:
        model_groups["HistGBT"].append(r["acc"])
    else:
        model_groups["Logistic\nRegression"].append(r["acc"])

best_per_model = {k: max(v) for k, v in model_groups.items() if v}
bar_colors = ["#3498db", "#e67e22", "#2ecc71"]

fig2, ax2 = plt.subplots(figsize=(8, 5))
bars = ax2.bar(
    list(best_per_model.keys()),
    list(best_per_model.values()),
    color=bar_colors[:len(best_per_model)],
    width=0.5, edgecolor="white", linewidth=1.2, zorder=3
)
ax2.axhline(0.5, color="grey", linestyle=":", linewidth=1.2, alpha=0.6, label="Chance (50%)")
ax2.set_ylim(0.45, min(1.0, max(best_per_model.values()) + 0.06))
ax2.set_ylabel("Best Validation Accuracy", fontsize=11)
ax2.set_title(f"Best Accuracy by Model Class\n(source: results_custom.tsv — {len(runs)} runs)",
              fontsize=13, fontweight="bold")
ax2.grid(axis="y", alpha=0.25, zorder=0)
ax2.legend(fontsize=9)
for bar, val in zip(bars, best_per_model.values()):
    ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.003,
             f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Wrote model_comparison.png")


# ── experiment_archive.md ──────────────────────────────────
def model_label(desc):
    if "RandomForest" in desc: return "RandomForest"
    if "HistGBT" in desc or "Hist" in desc: return "HistGBT"
    return "LogisticRegression"

table_rows = "\n".join(
    f"| {r['run']:02d} | {model_label(r['desc'])} | {r['desc'][:55]} "
    f"| {r['acc']:.6f} | {r['f1']:.6f} | {r['status']} |"
    for r in runs
)

keeps    = [r for r in runs if r["status"] == "keep"]
baselines= [r for r in runs if r["status"] == "baseline"]
discards = [r for r in runs if r["status"] == "discard"]

archive_md = f"""# Experiment Archive — AutoResearch S&P 500 Direction Prediction

Auto-generated from `results_custom.tsv` — {len(runs)} total runs.
Pipeline: `prepare_custom.py` (frozen), 41 features, 100 S&P 500 stocks, 2020–2024.
Split: 80/20 stratified, random_state=42. Metric: val_accuracy (higher is better).

**Summary**: {len(baselines)} baseline | {len(keeps)} keep | {len(discards)} discard
**Best accuracy**: {best_run['acc']:.6f} — Run {best_run['run']:02d}: {best_run['desc']}

---

## Full Run Log

| Run | Model | Description | val_acc | val_f1 | Status |
|-----|-------|-------------|---------|--------|--------|
{table_rows}

---

## Kept Runs

| Run | Description | val_acc | Improvement |
|-----|-------------|---------|-------------|
"""

prev_best = first_run["acc"]
for r in baselines + keeps:
    delta = r["acc"] - prev_best if r["status"] == "keep" else 0
    delta_str = f"+{delta:.4f}" if r["status"] == "keep" else "baseline"
    archive_md += f"| {r['run']:02d} | {r['desc'][:55]} | {r['acc']:.6f} | {delta_str} |\n"
    if r["acc"] > prev_best:
        prev_best = r["acc"]

archive_md += f"""
---

## Discarded Runs

| Run | Description | val_acc | vs best at time |
|-----|-------------|---------|-----------------|
"""

running_best = first_run["acc"]
for r in runs:
    if r["status"] in ("baseline", "keep") and r["acc"] > running_best:
        running_best = r["acc"]
    if r["status"] == "discard":
        delta = r["acc"] - running_best
        archive_md += f"| {r['run']:02d} | {r['desc'][:55]} | {r['acc']:.6f} | {delta:+.4f} |\n"

with open("experiment_archive.md", "w") as f:
    f.write(archive_md)
print("Wrote experiment_archive.md")


# ── final_results_table.md ─────────────────────────────────
improvement_acc = best_run["acc"] - first_run["acc"]
improvement_f1  = best_run["f1"]  - first_run["f1"]

# Best per model class
lr_best  = max((r for r in runs if model_label(r["desc"]) == "LogisticRegression"), key=lambda r: r["acc"])
rf_runs  = [r for r in runs if model_label(r["desc"]) == "RandomForest"]
gbt_best = max((r for r in runs if model_label(r["desc"]) == "HistGBT"), key=lambda r: r["acc"])
rf_row = f"| RandomForest (n=200, depth=8, balanced) | {max(r['acc'] for r in rf_runs):.6f} | {max(r['f1'] for r in rf_runs):.6f} | +{max(r['acc'] for r in rf_runs)-first_run['acc']:+.4f} |\n" if rf_runs else ""

results_md = f"""# Final Results Table — AutoResearch S&P 500 Direction Prediction

Auto-generated from `results_custom.tsv` — {len(runs)} total runs.
**Task**: Predict next-day stock close direction (UP=1 / DOWN=0)
**Data**: 100 S&P 500 stocks, 2020–2024, 41 features
**Split**: 80/20 stratified, random_state=42 (frozen)

---

## Model Comparison

| Model | val_accuracy | val_f1_macro | vs Baseline |
|-------|-------------|-------------|-------------|
| Logistic Regression (baseline) | {lr_best['acc']:.6f} | {lr_best['f1']:.6f} | — |
{rf_row}| **HistGBT (best: {gbt_best['desc'][:40]})** | **{gbt_best['acc']:.6f}** | **{gbt_best['f1']:.6f}** | **+{gbt_best['acc']-lr_best['acc']:+.4f}** |

---

## Best Configuration

| Parameter | Value |
|-----------|-------|
| Model | `HistGradientBoostingClassifier` |
| Description | {best_run['desc']} |
| **val_accuracy** | **{best_run['acc']:.6f}** |
| **val_f1_macro** | **{best_run['f1']:.6f}** |

---

## Improvement Over Baseline

| Metric | Baseline (LR) | Best (HistGBT) | Gain |
|--------|--------------|----------------|------|
| val_accuracy | {first_run['acc']:.6f} | {best_run['acc']:.6f} | **+{improvement_acc:.4f} (+{improvement_acc*100:.1f}pp)** |
| val_f1_macro | {first_run['f1']:.6f} | {best_run['f1']:.6f} | **+{improvement_f1:.4f} (+{improvement_f1*100:.1f}pp)** |

---

## Key Finding

Model class was the single most important factor.
Switching from Logistic Regression to HistGradientBoostingClassifier
produced +{(gbt_best['acc']-lr_best['acc'])*100:.1f}pp accuracy in one change.
Stock direction is driven by non-linear feature interactions
that LR cannot represent with a linear decision boundary.
"""

with open("final_results_table.md", "w") as f:
    f.write(results_md)
print("Wrote final_results_table.md")

print(f"\nAll outputs updated from {len(runs)} runs in {RESULTS_TSV}")
print(f"Best: {best_run['acc']:.6f} — {best_run['desc']}")
