"""
autonomous_loop.py — AutoResearch Autonomous Experiment Loop
============================================================
Run with:
    python autonomous_loop.py

Structure (matches the Karpathy AutoResearch framework):

  ┌─────────────────────────────────────┐
  │  INIT (once)                        │
  │  1. Read program.md                 │
  │  2. Run baseline → establish best   │
  └─────────────────────────────────────┘
               ↓
  ┌─────────────────────────────────────┐
  │  LOOP (≥ 6 iterations)              │
  │  1. Edit model.py (one change)      │
  │  2. python run_custom.py "<desc>"   │
  │  3. val_accuracy > best?            │
  │     yes → Keep,  update best        │
  │     no  → Revert model.py           │
  └─────────────────────────────────────┘
               ↓
  ┌─────────────────────────────────────┐
  │  FINALIZE (once)                    │
  │  python generate_charts.py          │
  │  results_custom.tsv → all outputs   │
  └─────────────────────────────────────┘
"""

import os, re, subprocess, textwrap, time
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────
DIR             = os.path.dirname(os.path.abspath(__file__))
MODEL_PY        = os.path.join(DIR, "model.py")
RESULTS_TSV     = os.path.join(DIR, "results_custom.tsv")
PROGRAM_MD      = os.path.join(DIR, "program.md")
REPORT_MD       = os.path.join(DIR, "autonomous_report.md")
GENERATE_CHARTS = os.path.join(DIR, "generate_charts.py")
RUN_TIMEOUT     = 240   # seconds before a run is declared a crash

# ── Search space ───────────────────────────────────────────
# Add new candidates here to extend the search.
# description must be unique — used to detect already-tried configs.
SEARCH_SPACE = [
    (
        "HistGBT depth=6 lr=0.05 iter=500 balanced [auto]",
        """\
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", HistGradientBoostingClassifier(
            max_iter=500, max_depth=6, learning_rate=0.05,
            class_weight="balanced", random_state=42,
        )),
    ])
"""
    ),
    (
        "HistGBT depth=6 lr=0.10 iter=500 balanced [auto]",
        """\
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", HistGradientBoostingClassifier(
            max_iter=500, max_depth=6, learning_rate=0.10,
            class_weight="balanced", random_state=42,
        )),
    ])
"""
    ),
    (
        "HistGBT depth=6 lr=0.05 iter=800 balanced [auto]",
        """\
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", HistGradientBoostingClassifier(
            max_iter=800, max_depth=6, learning_rate=0.05,
            class_weight="balanced", random_state=42,
        )),
    ])
"""
    ),
    (
        "HistGBT depth=6 lr=0.10 iter=800 balanced [auto]",
        """\
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", HistGradientBoostingClassifier(
            max_iter=800, max_depth=6, learning_rate=0.10,
            class_weight="balanced", random_state=42,
        )),
    ])
"""
    ),
    (
        "HistGBT depth=6 lr=0.05 iter=500 l2=0.5 balanced [auto]",
        """\
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", HistGradientBoostingClassifier(
            max_iter=500, max_depth=6, learning_rate=0.05,
            l2_regularization=0.5,
            class_weight="balanced", random_state=42,
        )),
    ])
"""
    ),
    (
        "HistGBT depth=6 lr=0.05 iter=500 min_samples=30 balanced [auto]",
        """\
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", HistGradientBoostingClassifier(
            max_iter=500, max_depth=6, learning_rate=0.05,
            min_samples_leaf=30,
            class_weight="balanced", random_state=42,
        )),
    ])
"""
    ),
    (
        "HistGBT depth=5 lr=0.05 iter=500 balanced [auto]",
        """\
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", HistGradientBoostingClassifier(
            max_iter=500, max_depth=5, learning_rate=0.05,
            class_weight="balanced", random_state=42,
        )),
    ])
"""
    ),
    (
        "HistGBT depth=6 lr=0.08 iter=600 balanced [auto]",
        """\
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", HistGradientBoostingClassifier(
            max_iter=600, max_depth=6, learning_rate=0.08,
            class_weight="balanced", random_state=42,
        )),
    ])
"""
    ),
    (
        "RandomForest n=300 depth=10 balanced [auto]",
        """\
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=300, max_depth=10,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )),
    ])
"""
    ),
    (
        "HistGBT depth=6 lr=0.05 iter=500 max_bins=128 balanced [auto]",
        """\
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", HistGradientBoostingClassifier(
            max_iter=500, max_depth=6, learning_rate=0.05,
            max_bins=128,
            class_weight="balanced", random_state=42,
        )),
    ])
"""
    ),
    (
        "HistGBT depth=7 lr=0.05 iter=500 balanced [auto]",
        """\
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", HistGradientBoostingClassifier(
            max_iter=500, max_depth=7, learning_rate=0.05,
            class_weight="balanced", random_state=42,
        )),
    ])
"""
    ),
    (
        "HistGBT depth=6 lr=0.10 iter=1000 balanced [auto]",
        """\
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", HistGradientBoostingClassifier(
            max_iter=1000, max_depth=6, learning_rate=0.10,
            class_weight="balanced", random_state=42,
        )),
    ])
"""
    ),
    # ── Extended search space (added after first two blocks) ──
    (
        "HistGBT depth=6 lr=0.10 iter=1200 balanced [auto]",
        """\
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", HistGradientBoostingClassifier(
            max_iter=1200, max_depth=6, learning_rate=0.10,
            class_weight="balanced", random_state=42,
        )),
    ])
"""
    ),
    (
        "HistGBT depth=6 lr=0.10 iter=1500 balanced [auto]",
        """\
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", HistGradientBoostingClassifier(
            max_iter=1500, max_depth=6, learning_rate=0.10,
            class_weight="balanced", random_state=42,
        )),
    ])
"""
    ),
    (
        "HistGBT depth=6 lr=0.10 iter=1000 l2=0.1 balanced [auto]",
        """\
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", HistGradientBoostingClassifier(
            max_iter=1000, max_depth=6, learning_rate=0.10,
            l2_regularization=0.1,
            class_weight="balanced", random_state=42,
        )),
    ])
"""
    ),
    (
        "HistGBT depth=6 lr=0.10 iter=1000 min_samples=20 balanced [auto]",
        """\
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", HistGradientBoostingClassifier(
            max_iter=1000, max_depth=6, learning_rate=0.10,
            min_samples_leaf=20,
            class_weight="balanced", random_state=42,
        )),
    ])
"""
    ),
    (
        "HistGBT depth=7 lr=0.10 iter=1000 balanced [auto]",
        """\
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", HistGradientBoostingClassifier(
            max_iter=1000, max_depth=7, learning_rate=0.10,
            class_weight="balanced", random_state=42,
        )),
    ])
"""
    ),
    (
        "HistGBT depth=6 lr=0.05 iter=1000 balanced [auto]",
        """\
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", HistGradientBoostingClassifier(
            max_iter=1000, max_depth=6, learning_rate=0.05,
            class_weight="balanced", random_state=42,
        )),
    ])
"""
    ),
]


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def read_model():
    with open(MODEL_PY) as f:
        return f.read()

def write_model(content):
    with open(MODEL_PY, "w") as f:
        f.write(content)

def already_tried(description):
    if not os.path.exists(RESULTS_TSV):
        return False
    with open(RESULTS_TSV) as f:
        return description in f.read()

def read_best_from_tsv():
    """Return (best_acc, best_desc) from baseline + keep rows in results_custom.tsv."""
    if not os.path.exists(RESULTS_TSV):
        return 0.0, "none"
    best_acc, best_desc = 0.0, "none"
    with open(RESULTS_TSV) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            try:
                acc    = float(parts[1])
                status = parts[3]
                desc   = parts[4]
                if status in ("baseline", "keep") and acc > best_acc:
                    best_acc, best_desc = acc, desc
            except ValueError:
                continue
    return best_acc, best_desc

def fix_last_tsv_status(new_status):
    """Correct the status of the last row in results_custom.tsv."""
    if not os.path.exists(RESULTS_TSV):
        return
    with open(RESULTS_TSV) as f:
        lines = f.readlines()
    if len(lines) < 2:
        return
    parts = lines[-1].rstrip("\n").split("\t")
    if len(parts) >= 4:
        parts[3] = new_status
        lines[-1] = "\t".join(parts) + "\n"
    with open(RESULTS_TSV, "w") as f:
        f.writelines(lines)

def run_cmd(description, flag="--discard"):
    """Run run_custom.py and return (val_acc, val_f1, elapsed, status, stderr)."""
    cmd = f'python run_custom.py "{description}" {flag}'
    start = time.time()
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True,
                           text=True, cwd=DIR, timeout=RUN_TIMEOUT)
        elapsed = time.time() - start
        acc_m = re.search(r"val_accuracy[:\s]+([\d.]+)", r.stdout)
        f1_m  = re.search(r"val_f1[_macro]*[:\s]+([\d.]+)", r.stdout)
        val_acc = float(acc_m.group(1)) if acc_m else None
        val_f1  = float(f1_m.group(1))  if f1_m  else None
        if val_acc is None:
            return None, None, elapsed, "error", r.stderr.strip()[:300]
        return val_acc, val_f1, elapsed, "ok", ""
    except subprocess.TimeoutExpired:
        return None, None, time.time()-start, "timeout", f"Timed out after {RUN_TIMEOUT}s"
    except Exception as e:
        return None, None, time.time()-start, "error", str(e)


# ══════════════════════════════════════════════════════════════
# PHASE 1 — INIT
# ══════════════════════════════════════════════════════════════

def init():
    """Read program.md for rules, then establish the current best accuracy."""

    print("┌─────────────────────────────────────┐")
    print("│  INIT                               │")
    print("└─────────────────────────────────────┘")

    # Step 1: Read program.md
    if os.path.exists(PROGRAM_MD):
        with open(PROGRAM_MD) as f:
            lines = f.read().splitlines()
        print("\n  program.md — objective:")
        for line in lines[:12]:
            if line.strip():
                print(f"  {line}")
    else:
        print("  WARNING: program.md not found.")

    # Step 2: Establish best from existing results or run baseline
    best_acc, best_desc = read_best_from_tsv()

    if best_acc == 0.0:
        # No runs yet — run the current model.py as the baseline
        print("\n  No existing results found. Running baseline ...")
        desc = "baseline [auto-init]"
        val_acc, val_f1, elapsed, status, err = run_cmd(desc, flag="--baseline")
        if val_acc is not None:
            best_acc, best_desc = val_acc, desc
            print(f"  Baseline established: val_accuracy = {best_acc:.6f}")
        else:
            print(f"  ERROR: baseline run failed — {err}")
    else:
        print(f"\n  Best from results_custom.tsv : {best_acc:.6f}")
        print(f"  Config                       : {best_desc}")

    print()
    return best_acc, best_desc


# ══════════════════════════════════════════════════════════════
# PHASE 2 — LOOP
# ══════════════════════════════════════════════════════════════

def loop(best_acc, best_desc, n=6):
    """Run n experiments. Edit model.py → eval → keep or revert."""

    print("┌─────────────────────────────────────┐")
    print(f"│  LOOP  ({n} iterations)               │")
    print("└─────────────────────────────────────┘\n")

    # Pick next n untried candidates
    candidates = [(d, b) for d, b in SEARCH_SPACE if not already_tried(d)]
    if not candidates:
        print("  All search space candidates already tried.")
        print("  Add new entries to SEARCH_SPACE to continue.\n")
        return best_acc, best_desc, []

    if len(candidates) < n:
        print(f"  Only {len(candidates)} untried candidates available (wanted {n}).")
    candidates = candidates[:n]

    checkpoint = read_model()   # current best model.py content
    records = []

    for i, (desc, body) in enumerate(candidates, 1):
        print(f"  [{i}/{len(candidates)}] Edit model.py → {desc}")

        # ── Edit model.py ──────────────────────────────
        write_model(body)

        # ── python run_custom.py "<desc>" ──────────────
        val_acc, val_f1, elapsed, status, err = run_cmd(desc, flag="--discard")

        # ── val_accuracy > best? ───────────────────────
        if status != "ok" or val_acc is None:
            outcome = "CRASH"
            reason  = err
            write_model(checkpoint)          # revert

        elif val_acc > best_acc:
            outcome    = "KEEP"
            reason     = f"+{val_acc - best_acc:.4f} over best ({best_acc:.6f})"
            best_acc   = val_acc
            best_desc  = desc
            checkpoint = body                # update checkpoint
            fix_last_tsv_status("keep")      # correct TSV: discard → keep

        else:
            outcome = "REVERT"
            reason  = f"{val_acc - best_acc:+.4f} vs best ({best_acc:.6f}) — model.py reverted"
            write_model(checkpoint)          # revert

        # ── Print result ───────────────────────────────
        acc_str = f"{val_acc:.6f}" if val_acc else "n/a"
        f1_str  = f"{val_f1:.6f}"  if val_f1  else "n/a"
        icon    = "✅" if outcome == "KEEP" else ("💥" if outcome == "CRASH" else "↩️ ")
        print(f"         val_accuracy={acc_str}  val_f1={f1_str}")
        print(f"         {icon} {outcome} — {reason}  ({elapsed:.1f}s)\n")

        records.append({
            "index": i, "desc": desc,
            "val_acc": val_acc, "val_f1": val_f1,
            "outcome": outcome, "reason": reason, "elapsed": elapsed,
        })

    return best_acc, best_desc, records


# ══════════════════════════════════════════════════════════════
# PHASE 3 — FINALIZE
# ══════════════════════════════════════════════════════════════

def finalize(best_acc, best_desc, records):
    """Write report and regenerate all outputs from results_custom.tsv."""

    print("┌─────────────────────────────────────┐")
    print("│  FINALIZE                           │")
    print("└─────────────────────────────────────┘\n")

    # Write autonomous_report.md
    _write_report(records, best_acc, best_desc)
    print(f"  autonomous_report.md   updated")

    # Regenerate charts + archive + results table
    r = subprocess.run(f"python {GENERATE_CHARTS}", shell=True,
                       capture_output=True, text=True, cwd=DIR)
    if r.returncode == 0:
        for line in r.stdout.strip().splitlines():
            print(f"  {line}")
    else:
        print(f"  WARNING: generate_charts.py failed — {r.stderr.strip()[:200]}")

    print(f"\n  Final best accuracy : {best_acc:.6f}")
    print(f"  Config              : {best_desc}\n")


def _write_report(records, best_acc, best_desc):
    now   = datetime.now().strftime("%Y-%m-%d %H:%M")
    kept  = [r for r in records if r["outcome"] == "KEEP"]
    reverted = [r for r in records if r["outcome"] == "REVERT"]
    crashed  = [r for r in records if r["outcome"] == "CRASH"]

    def fmt_row(r):
        acc = f"{r['val_acc']:.6f}" if r["val_acc"] else "n/a"
        return f"| {r['index']} | {r['desc'][:50]} | {acc} | {r['outcome']} | {r['elapsed']:.1f}s |"

    rows = "\n        ".join(fmt_row(r) for r in records)

    kept_lines    = "\n".join(f"- **{r['desc']}** — {r['val_acc']:.6f} ({r['reason']})" for r in kept) or "_None_"
    revert_lines  = "\n".join(f"- {r['desc']} — {r['reason']}" for r in reverted) or "_None_"
    crash_lines   = "\n".join(f"- {r['desc']} — {r['reason']}" for r in crashed) or "_None_"

    content = f"""# Autonomous Loop Report
**Generated**: {now}
**Runs this block**: {len(records)}
**Final best accuracy**: {best_acc:.6f} — {best_desc}

---

## Outcome Table

| # | Description | val_acc | Outcome | Time |
|---|-------------|---------|---------|------|
        {rows}

---

## Kept (improved accuracy)
{kept_lines}

## Reverted (no improvement — model.py rolled back)
{revert_lines}

## Crashed / Timed Out
{crash_lines}
"""
    with open(REPORT_MD, "w") as f:
        f.write(content)


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  AutoResearch — Autonomous Experiment Loop")
    print("="*60 + "\n")

    best_acc, best_desc          = init()
    best_acc, best_desc, records = loop(best_acc, best_desc, n=6)
    finalize(best_acc, best_desc, records)
