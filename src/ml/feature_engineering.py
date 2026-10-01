"""
Feature engineering for batch/offline contexts.
These features are computed on a DataFrame window and are used
during EDA / future retraining on richer datasets.

NOTE: The production XGBoost model is trained on the Kaggle creditcard.csv
which already contains PCA-transformed V1-V28 features.
This module is provided for future retraining on raw transactional data.
"""

import pandas as pd
import numpy as np


class FeatureEngineer:

    def __init__(self, window_size: int = 5) -> None:
        self.window_size = window_size

    def add_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Transaction velocity in recent window
        df["transaction_velocity"] = (
            df["Amount"]
            .rolling(window=self.window_size, min_periods=1)
            .count()
        )

        # Average amount in recent transactions
        df["avg_transaction_amount"] = (
            df["Amount"]
            .rolling(window=self.window_size, min_periods=1)
            .mean()
        )

        # Deviation from rolling average
        df["amount_deviation"] = df["Amount"] - df["avg_transaction_amount"]

        # High amount flag — top 5% of amounts in the current batch
        threshold = df["Amount"].quantile(0.95)
        df["high_amount_flag"] = (df["Amount"] > threshold).astype(int)

        # Rapid-fire flag — more than 4 transactions in window
        df["rapid_fire_flag"] = (df["transaction_velocity"] > 4).astype(int)

        return df

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract cyclical time-of-day features from a Unix timestamp column."""
        df = df.copy()
        if "timestamp" in df.columns:
            df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
            df["day_of_week"] = pd.to_datetime(df["timestamp"]).dt.dayofweek
            # Cyclical encoding
            df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        return df
