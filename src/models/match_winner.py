from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelArtifacts:
    pipeline: Any
    feature_columns: list[str]
    model_name: str


def _build_logistic_pipeline(feature_columns: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_columns,
            )
        ],
        remainder="drop",
    )

    base_model = LogisticRegression(max_iter=2000, random_state=42)
    calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=3)

    return Pipeline([("pre", pre), ("model", calibrated)])


def train_match_winner_model(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_col: str,
    train_mask: pd.Series,
) -> ModelArtifacts:
    X_train = df.loc[train_mask, feature_columns]
    y_train = df.loc[train_mask, target_col].astype(int)

    pipe = _build_logistic_pipeline(feature_columns)
    pipe.fit(X_train, y_train)
    return ModelArtifacts(pipeline=pipe, feature_columns=feature_columns, model_name="logistic_calibrated")


def predict_proba(artifacts: ModelArtifacts, df: pd.DataFrame) -> np.ndarray:
    X = df[artifacts.feature_columns]
    return artifacts.pipeline.predict_proba(X)[:, 1]


def evaluate_binary_predictions(y_true: pd.Series, y_prob: np.ndarray) -> dict[str, float]:
    y_true_i = y_true.astype(int)
    metrics = {
        "log_loss": float(log_loss(y_true_i, y_prob, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true_i, y_prob)),
    }
    if len(set(y_true_i)) > 1:
        metrics["auc"] = float(roc_auc_score(y_true_i, y_prob))
    return metrics
