from src.features.indicators import FeatureEngineer
from src.models.xgboost_model import XGBoostModel
import pandas as pd
import numpy as np

class MLStrategy:
    def __init__(self, model_path='models/xgboost_btc.json'):
        self.model = XGBoostModel(model_path)
        self.feature_engineer = FeatureEngineer()
        self.data_buffer = []

    def on_candle(self, candle):
        # candle is [timestamp, open, high, low, close, volume]
        self.data_buffer.append(candle)
        if len(self.data_buffer) > 100: # Keep enough data for indicators
            self.data_buffer.pop(0)

        if len(self.data_buffer) < 30:
            return "HOLD"

        df = pd.DataFrame(self.data_buffer, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = self.feature_engineer.add_indicators(df)

        if df.empty:
            return "HOLD"

        # Last row features
        last_features = df.iloc[-1:].drop(['timestamp', 'target'], axis=1, errors='ignore')

        # Prediction
        try:
            prob = self.model.predict_proba(last_features)[0][1]

            if prob > 0.6: # Conviction threshold
                return "BUY"
            elif prob < 0.4:
                return "SELL"
            else:
                return "HOLD"
        except:
            return "HOLD"

    def train(self, historical_df):
        df = self.feature_engineer.add_indicators(historical_df)
        df.dropna(subset=['target'], inplace=True) # For training we need the target
        X = df.drop(['timestamp', 'target', 'close'], axis=1, errors='ignore') # Avoid data leakage
        y = df['target']
        self.model.train(X, y)
