from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def build_tree_pipeline(feature_columns: list[str]) -> tuple[str, Any]:
    pre = ColumnTransformer(
        [("num", SimpleImputer(strategy="median"), feature_columns)], remainder="drop"
    )

    try:
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
        )
        calibrated = CalibratedClassifierCV(model, method="sigmoid", cv=3)
        return "xgboost_calibrated", Pipeline([("pre", pre), ("model", calibrated)])
    except Exception:
        return "unavailable", None


def train_tree_if_available(df: pd.DataFrame, feature_columns: list[str], target_col: str, train_mask: pd.Series):
    name, pipe = build_tree_pipeline(feature_columns)
    if pipe is None:
        return None
    X_train = df.loc[train_mask, feature_columns]
    y_train = df.loc[train_mask, target_col].astype(int)
    pipe.fit(X_train, y_train)
    return {"model_name": name, "pipeline": pipe, "feature_columns": feature_columns}
