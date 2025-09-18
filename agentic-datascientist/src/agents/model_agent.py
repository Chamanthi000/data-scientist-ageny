from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from ..utils.metrics import binary_metrics

class ModelAgent:
    def __init__(self, target: str):
        self.target = target

    def _split(self, df: pd.DataFrame):
        y = df[self.target].astype(int)
        X = df.drop(columns=[self.target])
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def _preprocess(self, X: pd.DataFrame):
        num_cols = X.select_dtypes(include='number').columns.tolist()
        cat_cols = [c for c in X.columns if c not in num_cols]
        numeric = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        categorical = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        pre = ColumnTransformer(
            transformers=[
                ('num', numeric, num_cols),
                ('cat', categorical, cat_cols)
            ]
        )
        return pre

    def train(self, df: pd.DataFrame, outdir: str | Path) -> Dict[str, Any]:
        outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
        X_train, X_test, y_train, y_test = self._split(df)
        pre = self._preprocess(X_train)

        candidates = {
            "random_forest": RandomForestClassifier(n_estimators=300, random_state=42),
            "xgboost": XGBClassifier(
                n_estimators=600, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                learning_rate=0.05, eval_metric='logloss', n_jobs=4, random_state=42
            ),
            "lightgbm": LGBMClassifier(
                n_estimators=700, max_depth=-1, num_leaves=63, subsample=0.8,
                colsample_bytree=0.8, learning_rate=0.05, random_state=42
            )
        }

        best_name, best_score, best_pipe = None, -1, None
        metrics_map = {}
        for name, model in candidates.items():
            pipe = Pipeline([('pre', pre), ('model', model)])
            pipe.fit(X_train, y_train)
            proba = pipe.predict_proba(X_test)[:, 1]
            mets = binary_metrics(y_test, proba)
            metrics_map[name] = mets
            if mets.get('roc_auc', 0) > best_score:
                best_name, best_score, best_pipe = name, mets.get('roc_auc', 0), pipe

        assert best_pipe is not None
        joblib.dump(best_pipe, outdir / "model.joblib")
        return {"best_model": best_name, "metrics": metrics_map}
