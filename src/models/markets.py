from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


class RegressionMarketModel:
    def __init__(self, feature_columns: list[str], target_col: str):
        self.feature_columns = feature_columns
        self.target_col = target_col
        self.pipeline = Pipeline(
            [
                (
                    "pre",
                    ColumnTransformer(
                        [("num", SimpleImputer(strategy="median"), self.feature_columns)],
                        remainder="drop",
                    ),
                ),
                ("model", LinearRegression()),
            ]
        )

    def fit(self, df: pd.DataFrame, train_mask: pd.Series):
        self.pipeline.fit(df.loc[train_mask, self.feature_columns], df.loc[train_mask, self.target_col])
        return self

    def predict(self, df: pd.DataFrame):
        return self.pipeline.predict(df[self.feature_columns])
