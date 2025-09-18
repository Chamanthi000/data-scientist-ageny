from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd

@dataclass
class DatasetSummary:
    n_rows: int
    n_cols: int
    columns: Dict[str, Dict[str, Any]]

class DatasetAgent:
    def summarize(self, df: pd.DataFrame) -> DatasetSummary:
        cols = {}
        for c in df.columns:
            s = df[c]
            info = {
                "dtype": str(s.dtype),
                "nulls": int(s.isna().sum()),
                "unique": int(s.nunique()),
            }
            if pd.api.types.is_numeric_dtype(s):
                info.update({
                    "mean": float(s.mean()),
                    "std": float(s.std()),
                    "min": float(s.min()),
                    "max": float(s.max()),
                })
            cols[c] = info
        return DatasetSummary(
            n_rows=int(df.shape[0]),
            n_cols=int(df.shape[1]),
            columns=cols,
        )
