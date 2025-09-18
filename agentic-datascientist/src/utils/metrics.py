from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def binary_metrics(y_true, y_pred_proba, threshold=0.5) -> Dict[str, float]:
    y_pred = (y_pred_proba >= threshold).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba))
    except Exception:
        pass
    return out
