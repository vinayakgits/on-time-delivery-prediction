"""XGBoost model: will an order be delivered on time?

Two jobs, one file, JSON output:
  train   : delivered orders (CSV)       -> saved model + metrics
  predict : new orders (CSV / JSON file / inline JSON) -> predictions

Usage:
  python src/xgboost_model.py train
  python src/xgboost_model.py train --data data/train/orders_train.csv
  python src/xgboost_model.py predict --data data/predict/orders_new.csv
  python src/xgboost_model.py predict --json data/predict/orders_new.json
  python src/xgboost_model.py predict --order "{\"order_id\": \"ORD1\", ...}"
  add --output results.json to save the JSON to a file
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN = ROOT / "data" / "train" / "orders_train.csv"
MODEL_FILE = ROOT / "models" / "xgb_model.json"
META_FILE = ROOT / "models" / "xgb_meta.json"
METRICS_FILE = ROOT / "outputs" / "xgb_metrics.json"

# ---- Input columns -------------------------------------------------------
ID_COL = "order_id"
TARGET_COL = "actual_delivery_date"          # training only
TEXT_COLS = ["courier_partner", "courier_service_type", "payment_mode", "product_category"]
PREDICT_COLS = [ID_COL, "order_datetime", "distance_km", *TEXT_COLS,
                "weight_g", "promised_delivery_date"]
TRAIN_COLS = PREDICT_COLS + [TARGET_COL]

# ---- Features the model uses (some are calculated) ----------------------
FEATURES = ["distance_km", "weight_g", "promised_days", "order_hour",
            "order_day_of_week", "is_sale_period", *TEXT_COLS]

# Sale / festival periods (inclusive). Edit these to match your business calendar.
SALE_PERIODS = [
    ("2025-11-24", "2025-12-02"),   # Black Friday / Cyber sale
    ("2025-12-20", "2025-12-31"),   # Year-end sale
    ("2026-01-18", "2026-01-26"),   # Republic Day sale
    ("2026-08-08", "2026-08-15"),   # Independence Day sale
    ("2026-10-01", "2026-11-10"),   # Festive season (Navratri to Diwali)
]

THRESHOLD = 0.5   # probability_on_time >= THRESHOLD -> "Yes"
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
LABELS = {
    "distance_km": "Distance (km)", "weight_g": "Weight (g)",
    "promised_days": "Promised days", "order_hour": "Order hour",
    "order_day_of_week": "Order day", "is_sale_period": "Sale period",
    "courier_partner": "Courier", "courier_service_type": "Service",
    "payment_mode": "Payment", "product_category": "Category",
}


class InputError(Exception):
    pass


def is_sale_period(dates):
    """True when the order date falls inside any SALE_PERIODS range."""
    day = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    flag = np.zeros(len(day), dtype=bool)
    for start, end in SALE_PERIODS:
        flag |= np.asarray((day >= pd.Timestamp(start)) & (day <= pd.Timestamp(end)))
    return flag


def build_features(df, categories=None):
    """Turn raw order columns into model features.

    categories=None (training): learn the allowed text values from the data.
    categories=dict (prediction): only accept values seen in training.
    Returns (features, problems, categories); problems maps row index -> list of messages.
    """
    problems = {}

    def flag(mask, message):
        for i in df.index[np.asarray(mask, dtype=bool)]:
            problems.setdefault(i, []).append(message if isinstance(message, str) else message(i))

    X = pd.DataFrame(index=df.index)
    order_dt = pd.to_datetime(df["order_datetime"], errors="coerce")
    promised = pd.to_datetime(df["promised_delivery_date"], errors="coerce")
    flag(order_dt.isna(), "order_datetime is missing or not a valid date-time")
    flag(promised.isna(), "promised_delivery_date is missing or not a valid date")

    for col in ["distance_km", "weight_g"]:
        X[col] = pd.to_numeric(df[col], errors="coerce")
        flag(~(X[col] > 0), f"{col} must be a number above 0")

    X["promised_days"] = (promised.dt.normalize() - order_dt.dt.normalize()).dt.days
    flag(X["promised_days"] < 0, "promised_delivery_date is before order_datetime")
    X["order_hour"] = order_dt.dt.hour
    X["order_day_of_week"] = order_dt.dt.day_of_week
    X["is_sale_period"] = is_sale_period(order_dt.fillna(pd.Timestamp(0))).astype(int)

    learned = {}
    for col in TEXT_COLS:
        raw = df[col].astype("string").str.strip().replace("", pd.NA)
        flag(raw.isna(), f"{col} is missing")
        if categories is None:
            lookup = {}
            for value in sorted(raw.dropna().unique()):
                lookup.setdefault(value.lower(), value)   # "courier a" -> "Courier A"
            cats = sorted(set(lookup.values()))
        else:
            cats = categories[col]
            lookup = {c.lower(): c for c in cats}
        value = raw.str.lower().map(lookup)
        flag(raw.notna() & value.isna(),
             lambda i, c=col, cs=cats: f"{c} '{raw[i]}' is unknown. Allowed: {cs}")
        X[col] = pd.Categorical(value, categories=cats)
        learned[col] = cats

    return X[FEATURES], problems, learned


def check_columns(df, required):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise InputError(f"Missing columns: {missing}. Required: {required}")


# ---- Metrics (no scikit-learn) -------------------------------------------
def roc_auc(y_true, prob):
    ranks = pd.Series(prob).rank().to_numpy()
    pos = np.asarray(y_true) == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def evaluate(y_true, prob):
    y_true = np.asarray(y_true)
    pred = (prob >= THRESHOLD).astype(int)
    cm = [[int(((y_true == a) & (pred == p)).sum()) for p in (0, 1)] for a in (0, 1)]
    per_class = {}
    for label, name in [(0, "late"), (1, "on_time")]:
        tp, predicted, actual = cm[label][label], cm[0][label] + cm[1][label], sum(cm[label])
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"precision": round(precision, 4), "recall": round(recall, 4),
                           "f1": round(f1, 4), "support": actual}
    majority = max(y_true.mean(), 1 - y_true.mean())
    return {
        "accuracy": round(float((y_true == pred).mean()), 4),
        "roc_auc": round(float(roc_auc(y_true, prob)), 4),
        "baseline_accuracy": round(float(majority), 4),
        **per_class,
        "confusion_matrix": {"rows_actual_cols_predicted": ["Late", "On time"], "matrix": cm},
    }


# ---- Train ---------------------------------------------------------------
def train(data_path):
    df = pd.read_csv(data_path, encoding="utf-8-sig", dtype={ID_COL: "string"})
    check_columns(df, TRAIN_COLS)

    X, problems, categories = build_features(df)
    order_dt = pd.to_datetime(df["order_datetime"], errors="coerce")
    actual = pd.to_datetime(df[TARGET_COL], errors="coerce")
    promised = pd.to_datetime(df["promised_delivery_date"], errors="coerce")
    for i in df.index[actual.isna()]:
        problems.setdefault(i, []).append(f"{TARGET_COL} is missing or not a valid date")
    for i in df.index[actual.dt.normalize() < order_dt.dt.normalize()]:
        problems.setdefault(i, []).append(f"{TARGET_COL} is before order_datetime")

    y = (actual.dt.normalize() <= promised.dt.normalize()).astype(int)   # 1 = on time
    keep = ~df.index.isin(list(problems))
    X, y, order_dt = X[keep], y[keep], order_dt[keep]
    if len(X) < 200:
        raise InputError(f"Only {len(X)} valid rows; at least 200 are needed to train")
    if y.nunique() < 2:
        raise InputError("Training data has only on-time or only late orders; both are needed")

    # Time-based split: train on older orders, test on the newest 20%
    order = np.argsort(order_dt.to_numpy(), kind="stable")
    X, y, order_dt = X.iloc[order], y.iloc[order], order_dt.iloc[order]
    test_start = int(len(X) * 0.8)
    val_start = int(test_start * 0.85)          # newest 15% of training rows -> early stopping
    X_fit, y_fit = X.iloc[:val_start], y.iloc[:val_start]
    X_val, y_val = X.iloc[val_start:test_start], y.iloc[val_start:test_start]
    X_test, y_test = X.iloc[test_start:], y.iloc[test_start:]

    model = xgb.XGBClassifier(
        n_estimators=1000, learning_rate=0.03, max_depth=4, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        enable_categorical=True, tree_method="hist", eval_metric="logloss",
        early_stopping_rounds=50, random_state=42, n_jobs=-1,
    )
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    prob = model.predict_proba(X_test)[:, 1]

    gain = model.get_booster().get_score(importance_type="gain")
    total = sum(gain.values()) or 1
    version = f"xgb_{datetime.now():%Y%m%d_%H%M%S}"
    fmt = lambda s: s.min().strftime("%Y-%m-%d") + " to " + s.max().strftime("%Y-%m-%d")

    result = {
        "status": "ok",
        "job": "train",
        "algorithm": "XGBoost",
        "model_version": version,
        "model_file": str(MODEL_FILE.relative_to(ROOT)),
        "data": {
            "total_rows": len(df),
            "rows_used": int(keep.sum()),
            "rows_skipped": len(problems),
            "skipped_examples": [
                {"row": int(i) + 2, "order_id": _clean(df.at[i, ID_COL]), "errors": msgs}
                for i, msgs in list(problems.items())[:5]
            ],
            "train_rows": test_start,
            "test_rows": len(X_test),
            "train_period": fmt(order_dt.iloc[:test_start]),
            "test_period": fmt(order_dt.iloc[test_start:]),
            "on_time_rate": round(float(y.mean()), 4),
        },
        "trees_used": model.best_iteration + 1,
        "metrics": evaluate(y_test, prob),
        "feature_importance": {
            k: round(v / total, 4) for k, v in sorted(gain.items(), key=lambda kv: -kv[1])
        },
    }

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_FILE)
    META_FILE.write_text(json.dumps({
        "model_version": version, "trained_at": datetime.now().isoformat(timespec="seconds"),
        "features": FEATURES, "categories": categories, "threshold": THRESHOLD,
    }, indent=2))
    METRICS_FILE.write_text(json.dumps(result, indent=2))
    return result


# ---- Predict -------------------------------------------------------------
def _clean(value):
    return None if pd.isna(value) else str(value)


def _reason(feature, value):
    if feature == "order_day_of_week":
        value = DAYS[int(value)]
    elif feature == "is_sale_period":
        value = "Yes" if value else "No"
    elif isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"{LABELS[feature]}: {value}"


def predict(df):
    if not MODEL_FILE.exists() or not META_FILE.exists():
        raise InputError("Model not found. Run: python src/xgboost_model.py train")
    meta = json.loads(META_FILE.read_text())
    check_columns(df, PREDICT_COLS)
    if df.empty:
        raise InputError("No orders given")

    X, problems, _ = build_features(df, meta["categories"])
    valid = X[~X.index.isin(list(problems))]

    results = {}
    if len(valid):
        booster = xgb.Booster()
        booster.load_model(MODEL_FILE)
        dm = xgb.DMatrix(valid, enable_categorical=True)
        prob = booster.predict(dm)
        contribs = booster.predict(dm, pred_contribs=True)[:, :-1]   # last column = bias
        for (i, row), p, c in zip(valid.iterrows(), prob, contribs):
            on_time = p >= meta["threshold"]
            # features that pushed this order most toward "late"
            worst = [j for j in np.argsort(c) if c[j] < 0][:3]
            results[i] = {
                "order_id": _clean(df.at[i, ID_COL]),
                "on_time": "Yes" if on_time else "No",
                "probability_on_time": round(float(p), 4),
                "confidence": round(float(p if on_time else 1 - p), 4),
                "risk_level": "High" if p < 0.4 else "Medium" if p < 0.7 else "Low",
                "late_risk_factors": [_reason(FEATURES[j], row[FEATURES[j]]) for j in worst],
            }

    errors = [{"row": int(pos) + 1, "order_id": _clean(df.at[i, ID_COL]), "errors": problems[i]}
              for pos, i in enumerate(df.index) if i in problems]
    return {
        "status": "ok" if not errors else "partial" if results else "error",
        "job": "predict",
        "model_version": meta["model_version"],
        "total": len(df),
        "predicted": len(results),
        "failed": len(errors),
        "summary": {
            "on_time": sum(r["on_time"] == "Yes" for r in results.values()),
            "late": sum(r["on_time"] == "No" for r in results.values()),
            "high_risk": sum(r["risk_level"] == "High" for r in results.values()),
        },
        "predictions": [results[i] for i in df.index if i in results],
        "errors": errors,
    }


def load_orders(args):
    if args.data:
        return pd.read_csv(args.data, encoding="utf-8-sig", dtype={ID_COL: "string"})
    text = Path(args.json).read_text(encoding="utf-8-sig") if args.json else args.order
    orders = json.loads(text)
    if isinstance(orders, dict):
        orders = orders.get("orders", [orders])   # accepts {"orders": [...]} too
    return pd.DataFrame(orders, dtype=object)


def main():
    parser = argparse.ArgumentParser(description="XGBoost on-time delivery model")
    sub = parser.add_subparsers(dest="job", required=True)

    t = sub.add_parser("train", help="train and save the model")
    t.add_argument("--data", default=str(DEFAULT_TRAIN), help="CSV of delivered orders")

    p = sub.add_parser("predict", help="predict with the saved model")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="CSV file of new orders (bulk)")
    src.add_argument("--json", help="JSON file: a list of orders (bulk)")
    src.add_argument("--order", help="inline JSON: one order or a list of orders")

    for s in (t, p):
        s.add_argument("--output", help="also save the JSON result to this file")

    args = parser.parse_args()
    try:
        result = train(args.data) if args.job == "train" else predict(load_orders(args))
    except (InputError, FileNotFoundError, ValueError) as e:
        result = {"status": "error", "job": args.job, "message": str(e)}
    except Exception as e:   # unexpected: still answer in JSON so the caller can read it
        result = {"status": "error", "job": args.job,
                  "message": f"Unexpected error: {type(e).__name__}: {e}"}

    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)
    sys.exit(1 if result["status"] == "error" else 0)


if __name__ == "__main__":
    main()
