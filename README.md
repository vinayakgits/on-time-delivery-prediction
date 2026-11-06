# On-Time Delivery Prediction (XGBoost)

Predict whether an order will be **delivered by its promised date**, at the time the order is placed.

> Data files, trained models and outputs are not included in this repo.
> Put your training CSV at `data/train/orders_train.csv` (or pass any path with `--data`),
> using the columns below.

## Input columns

| # | Column | Type | Example | Train | Predict |
|---|---|---|---|---|---|
| 1 | `order_id` | text | ORD123456 | ✅ | ✅ |
| 2 | `order_datetime` | date-time | 2026-11-06 18:45 | ✅ | ✅ |
| 3 | `distance_km` | number > 0 | 1150 | ✅ | ✅ |
| 4 | `courier_partner` | text | Courier A | ✅ | ✅ |
| 5 | `courier_service_type` | text | Express / Surface | ✅ | ✅ |
| 6 | `payment_mode` | text | COD / Prepaid | ✅ | ✅ |
| 7 | `product_category` | text | Wall art | ✅ | ✅ |
| 8 | `weight_g` | number > 0 | 3200 | ✅ | ✅ |
| 9 | `promised_delivery_date` | date | 2026-11-11 | ✅ | ✅ |
| 10 | `actual_delivery_date` | date | 2026-11-13 | ✅ | ❌ |

Text values are matched ignoring case. At prediction time, text values must be ones that appeared in training. For example, a new courier needs retraining first.

### Calculated automatically

| Feature | From |
|---|---|
| `promised_days` | `promised_delivery_date` − order date |
| `order_hour`, `order_day_of_week` | `order_datetime` |
| `is_sale_period` | `order_datetime` + `SALE_PERIODS` in `src/xgboost_model.py` (edit to match your calendar) |
| **Target** `on_time` | 1 if `actual_delivery_date` ≤ `promised_delivery_date`, else 0 |

## How to use

```
pip install pandas numpy xgboost

# Train
python src/xgboost_model.py train
python src/xgboost_model.py train --data path/to/delivered_orders.csv

# Predict: bulk CSV, bulk JSON file, or inline JSON
python src/xgboost_model.py predict --data data/predict/orders_new.csv
python src/xgboost_model.py predict --json data/predict/orders_new.json
python src/xgboost_model.py predict --order "{\"order_id\": \"ORD1\", \"order_datetime\": \"2026-11-07 19:30\", ...}"

# Save the JSON result to a file as well
python src/xgboost_model.py predict --data data/predict/orders_new.csv --output outputs/predictions.json
```

A JSON file can be a list `[{...}, {...}]` or `{"orders": [{...}, {...}]}`.

## Output (always JSON)

| `status` | Meaning | Exit code |
|---|---|---|
| `ok` | Everything worked | 0 |
| `partial` | Some rows were predicted; bad rows are listed in `errors` | 0 |
| `error` | Nothing worked; see `message` (or `errors`) | 1 |

**Prediction** returns `predictions[]` with these fields for each order:

| Field | Meaning |
|---|---|
| `order_id` | The order's ID |
| `on_time` | Yes / No |
| `probability_on_time` | 0–1 |
| `confidence` | 0–1 |
| `risk_level` | High (< 0.4), Medium (< 0.7), Low |
| `late_risk_factors` | Up to 3 inputs that pushed this order toward "late" |

It also returns a `summary` (on time, late, high risk) and `errors[]` (row, order_id, reasons). `row` 1 is the first order.

**Training** returns: row counts, train and test periods, accuracy, ROC-AUC, precision, recall, F1, the confusion matrix and feature importance. The same JSON is saved to `outputs/xgb_metrics.json`.

## How training works

1. Check that all required columns are present. Rows with bad values are skipped and reported.
2. Calculate the features and the target.
3. **Split by time:** the oldest 80% of orders are for training and the newest 20% are for testing. The newest 15% of the training part decides when to stop adding trees (early stopping).
4. Train XGBoost. Text columns are used directly (native categories).
5. Save `models/xgb_model.json` and `models/xgb_meta.json` (version, categories, threshold), plus the metrics.

## Project files

| File | Purpose |
|---|---|
| `src/xgboost_model.py` | Train and predict (JSON in, JSON out) |
| `data/train/orders_train.csv` | Your training data (not in repo) |
| `data/predict/` | Your new orders to predict (not in repo) |
| `models/xgb_model.json`, `models/xgb_meta.json` | Created by `train` (not in repo) |
| `outputs/xgb_metrics.json` | Created by `train` (not in repo) |

## Data rules

- Use only information known when the order is placed. Nothing from after it, such as ratings or complaints.
- Include only completed deliveries; leave out cancelled and returned orders.
- Collect at least 6 months and 5,000 orders, ideally 12+ months covering all sale seasons.
- Retrain monthly, and whenever a new courier, service or category is added.
