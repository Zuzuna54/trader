import pandas as pd
import ta

class FeatureEngineer:
    @staticmethod
    def add_indicators(df):
        df = df.copy()
        # Trend
        df['ema_9'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
        df['ema_21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()

        # Momentum
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

        # Volatility
        bb = ta.volatility.BollingerBands(df['close'])
        df['bb_high'] = bb.bollinger_hband()
        df['bb_mid'] = bb.bollinger_mavg()
        df['bb_low'] = bb.bollinger_lband()

        # Others
        df['roc'] = ta.momentum.ROCIndicator(df['close'], window=10).roc()

        # Returns
        df['returns'] = df['close'].pct_change()

        # Target: 1 if next close > current close, else 0
        # Only added during training
        df['target'] = (df['close'].shift(-1) > df['close']).astype(float)

        # Don't drop the very last row if we're doing inference (it will have NaN target but valid indicators)
        # Drop rows where indicators are NaN (the beginning of the df)
        df.dropna(subset=['ema_9', 'rsi', 'bb_high', 'roc'], inplace=True)
        return df
