"""
Run one experiment with real price + interest rate data.

Usage:
    python run_custom.py "description"             # logs as status=keep
    python run_custom.py "description" --baseline  # logs as status=baseline
    python run_custom.py "description" --discard   # logs as status=discard

Requires:
    fred_rates.csv  in this folder (download from https://fred.stlouisfed.org)
"""
import sys
import time
import subprocess
from prepare_custom import load_data, evaluate, log_result


def get_git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "no-git"


def main():
    args = sys.argv[1:]
    status = "keep"
    description_parts = []
    for a in args:
        if a == "--baseline":
            status = "baseline"
        elif a == "--discard":
            status = "discard"
        else:
            description_parts.append(a)
    description = " ".join(description_parts) if description_parts else "experiment"

    # 1. Load real data
    X_train, y_train, X_val, y_val, feature_names = load_data()
    up_pct = y_train.mean() * 100
    print(f"\nData: {X_train.shape[0]:,} train | {X_val.shape[0]:,} val | "
          f"{X_train.shape[1]} features")
    print(f"Features: {feature_names}")
    print(f"Label balance: {up_pct:.1f}% UP days in training set")

    # 2. Build model
    from model import build_model
    model = build_model()
    print(f"Model: {model}")

    # 3. Train
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"Training time: {train_time:.2f}s")

    # 4. Evaluate
    val_acc, val_f1 = evaluate(model, X_val, y_val)
    print(f"val_accuracy: {val_acc:.6f}")
    print(f"val_f1_macro: {val_f1:.6f}")

    # 5. Log to results_custom.tsv
    commit = get_git_hash()
    log_result(commit, val_acc, val_f1, status, description)
    print(f"Result logged to results_custom.tsv (status={status})")


if __name__ == "__main__":
    main()
