from src.data.fetcher import DataFetcher
from src.strategy.ml_strategy import MLStrategy
from src.execution.binance_executor import BinanceExecutor
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)

def run_backtest(symbol='BTC/USDT', interval='1h', start_year=2025):
    fetcher = DataFetcher()
    df = fetcher.fetch_historical_ohlcv(symbol, interval, start_year)

    # Split for training and testing
    train_size = int(len(df) * 0.7)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    strategy = MLStrategy()
    strategy.train(train_df)

    executor = BinanceExecutor(mock=True)

    logging.info("Starting Backtest...")
    for index, row in test_df.iterrows():
        candle = [row['timestamp'], row['open'], row['high'], row['low'], row['close'], row['volume']]
        signal = strategy.on_candle(candle)
        current_price = row['close']

        executor.check_risk_management(current_price)
        executor.execute_signal(signal, current_price)

    final_value = executor.balance_usdt + (executor.balance_btc * test_df.iloc[-1]['close'])
    logging.info(f"Backtest Complete. Final Portfolio Value: {final_value:.2f} USDT (Initial: 1000.00)")

if __name__ == "__main__":
    run_backtest()
