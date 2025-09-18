# EDA Report
- Rows: 400
- Cols: 6
## Columns:
- **age** (int64), nulls=0, unique=57
- **tenure_months** (int64), nulls=0, unique=59
- **monthly_spend** (float64), nulls=0, unique=395
- **segment** (object), nulls=0, unique=3
- **country** (object), nulls=0, unique=4
- **churned** (int64), nulls=0, unique=2

## Modeling
{
  "best_model": "xgboost",
  "metrics": {
    "random_forest": {
      "accuracy": 0.6125,
      "f1": 0.6593406593406593,
      "roc_auc": 0.6663510101010102
    },
    "xgboost": {
      "accuracy": 0.5875,
      "f1": 0.6206896551724138,
      "roc_auc": 0.6666666666666666
    },
    "lightgbm": {
      "accuracy": 0.625,
      "f1": 0.6666666666666666,
      "roc_auc": 0.6483585858585857
    }
  }
}