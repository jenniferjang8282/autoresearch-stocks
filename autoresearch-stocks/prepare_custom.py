"""
Custom data pipeline — real prices (yfinance) + real interest rates (FRED CSV).

Features (41 total):
  Original 13 : ret_1/3/5/10/20, vol_5/10, log_vol, vol_ratio,
                price_pos, rsi_like, macd, body_ratio
  Added previously (2) : ma_5_ratio, rate
  New — no extra data needed (16):
                ret_2, vol_20,
                ma_10_ratio, ma_20_ratio, ma_5_vs_20,
                intraday_ret, overnight_gap, hl_range,
                close_to_high, close_to_low,
                vol_change_1d, vol_ratio_20, vol_volatility,
                up_yesterday, up_days_10, streak
  New — SPY & VIX via yfinance (10):
                spy_ret_1/5/20, spy_vol_10,
                rel_ret_1/5/20, rel_vol_10,
                vix_level, vix_change

Before running:
  Save your FRED CSV in this folder as fred_rates.csv
  (download from https://fred.stlouisfed.org — e.g. FEDFUNDS or DGS10)
"""

import numpy as np
import pandas as pd
import csv as _csv
import os
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# ── Config ─────────────────────────────────────────────────
RANDOM_SEED   = 42
VAL_FRACTION  = 0.2
LOOKBACK_DAYS = 20
FRED_CSV      = "fred_rates.csv"
RESULTS_FILE  = "results_custom.tsv"
START_DATE    = "2020-01-01"
END_DATE      = "2024-01-01"

TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "JPM", "JNJ", "V",
    "UNH",  "XOM",  "PG",   "MA",    "HD",   "CVX",  "MRK",  "LLY", "ABBV","PEP",
    "KO",   "AVGO", "COST", "WMT",   "BAC",  "MCD",  "TMO",  "CSCO","ACN", "ABT",
    "CRM",  "NKE",  "NEE",  "LIN",   "TXN",  "DHR",  "BMY",  "PM",  "UPS", "MS",
    "RTX",  "AMGN", "ORCL", "INTC",  "QCOM", "HON",  "IBM",  "GS",  "CAT", "BLK",
    "SBUX", "INTU", "AXP",  "BA",    "LOW",  "MDLZ", "DE",   "SPGI","MMM", "GE",
    "ISRG", "AMT",  "GILD", "CVS",   "MO",   "ZTS",  "CI",   "SYK", "CB",  "TJX",
    "DUK",  "SO",   "PLD",  "WM",    "ECL",  "NOC",  "LMT",  "GD",  "ADP", "PAYX",
    "CTAS", "KLAC", "LRCX", "MCHP",  "ADI",  "AMAT", "SNPS", "CDNS","FTNT","PANW",
    "CRWD", "DDOG", "NOW",  "WDAY",  "TEAM", "CCI",  "EQIX", "PSA", "BRK-B","SNOW",
]


# ── 1. FRED interest rates ─────────────────────────────────
def _load_fred_rates(csv_path=FRED_CSV):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"\nFRED CSV not found: '{csv_path}'\n"
            "Download from https://fred.stlouisfed.org and save as fred_rates.csv\n"
        )
    df = pd.read_csv(csv_path)
    date_col, rate_col = df.columns[0], df.columns[1]
    df[rate_col] = pd.to_numeric(df[rate_col], errors="coerce")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    daily_idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    rates = df[rate_col].reindex(daily_idx).ffill() / 100.0
    print(f"  FRED '{rate_col}': {rates.index[0].date()} → {rates.index[-1].date()} "
          f"(range {rates.min():.4f}–{rates.max():.4f})")
    return rates


def _lookup_rate(rate_series, ts):
    """Look up rate for a timestamp, falling back to nearest earlier date."""
    if ts in rate_series.index:
        v = rate_series[ts]
    else:
        earlier = rate_series.index[rate_series.index <= ts]
        v = float(rate_series[earlier[-1]]) if len(earlier) > 0 else 0.0
    return 0.0 if np.isnan(v) else float(v)


# ── 2. SPY & VIX market data ───────────────────────────────
def _fetch_market_data(start=START_DATE, end=END_DATE):
    """
    Download SPY and ^VIX, return two DataFrames indexed by date with
    precomputed market-wide features.
    """
    print("  Downloading SPY and VIX …")

    spy_raw = yf.download("SPY", start=start, end=end,
                          auto_adjust=True, progress=False)
    vix_raw = yf.download("^VIX", start=start, end=end,
                          auto_adjust=True, progress=False)

    spy_close = spy_raw["Close"].squeeze().rename("close")
    vix_close = vix_raw["Close"].squeeze().rename("vix")

    # SPY features
    spy = pd.DataFrame(index=spy_close.index)
    spy["ret_1"]  = spy_close.pct_change(1)
    spy["ret_5"]  = spy_close.pct_change(5)
    spy["ret_20"] = spy_close.pct_change(20)
    log_ret       = np.log(spy_close / spy_close.shift(1))
    spy["vol_10"] = log_ret.rolling(10).std()

    # VIX features — reindex to SPY calendar and forward-fill gaps
    vix_aligned = vix_close.reindex(spy_close.index).ffill()
    spy["vix_level"]  = vix_aligned
    spy["vix_change"] = vix_aligned.pct_change(1)

    print(f"  SPY: {spy.index[0].date()} → {spy.index[-1].date()} "
          f"({spy.shape[0]} trading days)")
    return spy


def _lookup_market(mkt, ts):
    """Return market feature dict for a given date, or zeros if missing."""
    zero = {"spy_ret_1": 0.0, "spy_ret_5": 0.0, "spy_ret_20": 0.0,
            "spy_vol_10": 0.0, "vix_level": 0.0, "vix_change": 0.0}
    if ts not in mkt.index:
        earlier = mkt.index[mkt.index <= ts]
        if len(earlier) == 0:
            return zero
        ts = earlier[-1]
    row = mkt.loc[ts]
    return {k: (0.0 if np.isnan(row[v]) else float(row[v]))
            for k, v in [("spy_ret_1",  "ret_1"),
                          ("spy_ret_5",  "ret_5"),
                          ("spy_ret_20", "ret_20"),
                          ("spy_vol_10", "vol_10"),
                          ("vix_level",  "vix_level"),
                          ("vix_change", "vix_change")]}


# ── 3. Stock price download ────────────────────────────────
def _fetch_prices(tickers=TICKERS, start=START_DATE, end=END_DATE):
    print(f"  Downloading {len(tickers)} tickers ({start} → {end}) …")
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False)
    price_data = {}
    min_rows = LOOKBACK_DAYS + 2
    for ticker in tickers:
        try:
            df = pd.DataFrame({
                "date":   raw["Close"][ticker].index,
                "open":   raw["Open"][ticker].values,
                "high":   raw["High"][ticker].values,
                "low":    raw["Low"][ticker].values,
                "close":  raw["Close"][ticker].values,
                "volume": raw["Volume"][ticker].values,
            }).dropna()
            if len(df) >= min_rows:
                price_data[ticker] = df.reset_index(drop=True)
        except Exception:
            continue
    print(f"  {len(price_data)} tickers with sufficient data")
    return price_data


# ── 4. Streak helper ───────────────────────────────────────
def _streak(close, i, max_look=20):
    """
    Consecutive up/down streak ending at day i.
    Positive = up streak, negative = down streak.
    e.g. +3 means 3 consecutive up days; -2 means 2 consecutive down days.
    """
    if close[i] > close[i - 1]:
        direction = 1
    elif close[i] < close[i - 1]:
        direction = -1
    else:
        return 0
    count = 1
    for k in range(i - 1, max(i - max_look, 0), -1):
        if k == 0:
            break
        daily_dir = 1 if close[k] > close[k - 1] else -1
        if daily_dir == direction:
            count += 1
        else:
            break
    return direction * count


# ── 5. Feature engineering ────────────────────────────────
def _engineer_features(price_data, rate_series, mkt):
    out = []

    for ticker, g in price_data.items():
        g      = g.sort_values("date").reset_index(drop=True)
        dates  = g["date"].values
        close  = g["close"].values
        open_  = g["open"].values
        high   = g["high"].values
        low    = g["low"].values
        vol    = g["volume"].values
        n      = len(close)
        log_vol = np.log1p(vol)

        for i in range(LOOKBACK_DAYS, n - 1):

            # ── RETURNS ───────────────────────────────────
            ret_1  = (close[i] - close[i-1])  / (close[i-1]  + 1e-9)
            ret_2  = (close[i] - close[i-2])  / (close[i-2]  + 1e-9)   # NEW
            ret_3  = (close[i] - close[i-3])  / (close[i-3]  + 1e-9)
            ret_5  = (close[i] - close[i-5])  / (close[i-5]  + 1e-9)
            ret_10 = (close[i] - close[i-10]) / (close[i-10] + 1e-9)
            ret_20 = (close[i] - close[i-20]) / (close[i-20] + 1e-9)

            # ── VOLATILITY ────────────────────────────────
            vol_5  = np.std(np.diff(np.log(close[i-5:i+1]   + 1e-9)))
            vol_10 = np.std(np.diff(np.log(close[i-10:i+1]  + 1e-9)))
            vol_20 = np.std(np.diff(np.log(close[i-20:i+1]  + 1e-9)))  # NEW

            # ── MOVING AVERAGES ───────────────────────────
            ma_5  = np.mean(close[i-4:i+1])
            ma_10 = np.mean(close[i-9:i+1])
            ma_20 = np.mean(close[i-19:i+1])

            ma_5_ratio  = (ma_5  - close[i]) / (close[i] + 1e-9)
            ma_10_ratio = (ma_10 - close[i]) / (close[i] + 1e-9)       # NEW
            ma_20_ratio = (ma_20 - close[i]) / (close[i] + 1e-9)       # NEW
            ma_5_vs_20  = (ma_5  - ma_20)    / (close[i] + 1e-9)       # NEW

            # ── PRICE POSITION ────────────────────────────
            w_high    = np.max(high[i-20:i+1])
            w_low     = np.min(low[i-20:i+1])
            price_pos = (close[i] - w_low) / (w_high - w_low + 1e-9)

            # ── OHLCV INTRADAY ────────────────────────────
            intraday_ret  = (close[i] - open_[i])   / (open_[i]   + 1e-9)  # NEW
            overnight_gap = (open_[i] - close[i-1]) / (close[i-1] + 1e-9)  # NEW
            hl_range      = (high[i]  - low[i])     / (close[i]   + 1e-9)  # NEW
            close_to_high = (high[i]  - close[i])   / (close[i]   + 1e-9)  # NEW
            close_to_low  = (close[i] - low[i])     / (close[i]   + 1e-9)  # NEW
            body_ratio    = (close[i] - open_[i])   / (high[i] - low[i] + 1e-9)

            # ── MOMENTUM / OSCILLATORS ────────────────────
            diffs    = np.diff(close[i-14:i+1])
            rsi_like = np.sum(diffs > 0) / 14.0

            ema5  = pd.Series(close[i-20:i+1]).ewm(span=5,  adjust=False).mean().iloc[-1]
            ema20 = pd.Series(close[i-20:i+1]).ewm(span=20, adjust=False).mean().iloc[-1]
            macd  = (ema5 - ema20) / (close[i] + 1e-9)

            # ── VOLUME ────────────────────────────────────
            log_vol_today = log_vol[i]
            avg_vol_5     = np.mean(log_vol[i-5:i])
            avg_vol_20    = np.mean(log_vol[i-20:i])
            vol_ratio     = log_vol_today / (avg_vol_5  + 1e-9)
            vol_ratio_20  = log_vol_today / (avg_vol_20 + 1e-9)         # NEW
            vol_change_1d = (vol[i] - vol[i-1]) / (vol[i-1] + 1e-9)    # NEW
            vol_volatility = np.std(log_vol[i-5:i+1])                   # NEW

            # ── DIRECTION / STREAK ────────────────────────
            up_yesterday = int(close[i-1] > close[i-2])                 # NEW
            up_days_10   = int(np.sum(np.diff(close[i-10:i+1]) > 0))   # NEW
            streak       = _streak(close, i)                            # NEW

            # ── INTEREST RATE ─────────────────────────────
            cal_date = pd.Timestamp(dates[i])
            rate = _lookup_rate(rate_series, cal_date)

            # ── SPY / VIX MARKET FEATURES ─────────────────
            m = _lookup_market(mkt, cal_date)
            spy_ret_1  = m["spy_ret_1"]
            spy_ret_5  = m["spy_ret_5"]
            spy_ret_20 = m["spy_ret_20"]
            spy_vol_10 = m["spy_vol_10"]
            vix_level  = m["vix_level"]
            vix_change = m["vix_change"]

            # ── RELATIVE PERFORMANCE ──────────────────────
            rel_ret_1  = ret_1  - spy_ret_1
            rel_ret_5  = ret_5  - spy_ret_5
            rel_ret_20 = ret_20 - spy_ret_20
            rel_vol_10 = vol_10 - spy_vol_10

            # ── LABEL ─────────────────────────────────────
            label = 1 if close[i + 1] > close[i] else 0

            out.append({
                # original 13
                "ret_1":        ret_1,
                "ret_3":        ret_3,
                "ret_5":        ret_5,
                "ret_10":       ret_10,
                "ret_20":       ret_20,
                "vol_5":        vol_5,
                "vol_10":       vol_10,
                "log_vol":      log_vol_today,
                "vol_ratio":    vol_ratio,
                "price_pos":    price_pos,
                "rsi_like":     rsi_like,
                "macd":         macd,
                "body_ratio":   body_ratio,
                # previously added (2)
                "ma_5_ratio":   ma_5_ratio,
                "rate":         rate,
                # new — no extra data (16)
                "ret_2":        ret_2,
                "vol_20":       vol_20,
                "ma_10_ratio":  ma_10_ratio,
                "ma_20_ratio":  ma_20_ratio,
                "ma_5_vs_20":   ma_5_vs_20,
                "intraday_ret": intraday_ret,
                "overnight_gap":overnight_gap,
                "hl_range":     hl_range,
                "close_to_high":close_to_high,
                "close_to_low": close_to_low,
                "vol_change_1d":vol_change_1d,
                "vol_ratio_20": vol_ratio_20,
                "vol_volatility":vol_volatility,
                "up_yesterday": up_yesterday,
                "up_days_10":   up_days_10,
                "streak":       streak,
                # new — SPY/VIX (10)
                "spy_ret_1":    spy_ret_1,
                "spy_ret_5":    spy_ret_5,
                "spy_ret_20":   spy_ret_20,
                "spy_vol_10":   spy_vol_10,
                "rel_ret_1":    rel_ret_1,
                "rel_ret_5":    rel_ret_5,
                "rel_ret_20":   rel_ret_20,
                "rel_vol_10":   rel_vol_10,
                "vix_level":    vix_level,
                "vix_change":   vix_change,
                # label
                "label":        label,
            })

    feat_df = pd.DataFrame(out).dropna()
    X = feat_df.drop(columns=["label"]).values
    y = feat_df["label"].values
    feature_names = [c for c in feat_df.columns if c != "label"]
    return X, y, feature_names


# ── 6. Public API ──────────────────────────────────────────
def load_data(fred_csv=FRED_CSV):
    """
    Load all data sources, engineer 41 features, split train/val.
    Returns X_train, y_train, X_val, y_val, feature_names.
    """
    print("Loading FRED interest rate data …")
    rate_series = _load_fred_rates(fred_csv)

    print("Fetching market data (SPY + VIX) …")
    mkt = _fetch_market_data()

    print("Fetching stock prices …")
    price_data = _fetch_prices()

    print("Engineering features …")
    X, y, feature_names = _engineer_features(price_data, rate_series, mkt)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_FRACTION, random_state=RANDOM_SEED, stratify=y
    )
    return X_train, y_train, X_val, y_val, feature_names


# ── 7. Evaluation ──────────────────────────────────────────
def evaluate(model, X_val, y_val):
    y_pred = model.predict(X_val)
    return float(accuracy_score(y_val, y_pred)), float(f1_score(y_val, y_pred, average="macro"))


# ── 8. Logging ─────────────────────────────────────────────
def log_result(experiment_id, val_acc, val_f1, status, description):
    file_exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = _csv.writer(f, delimiter="\t")
        if not file_exists:
            writer.writerow(["experiment", "val_acc", "val_f1", "status", "description"])
        writer.writerow([experiment_id, f"{val_acc:.6f}", f"{val_f1:.6f}", status, description])
