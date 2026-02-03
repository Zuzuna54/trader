import logging
import signal
import sys
from src.config import Config
from src.data.streamer import DataStreamer
from src.strategy.ml_strategy import MLStrategy
from src.execution.binance_executor import BinanceExecutor
from src.data.fetcher import DataFetcher

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    logging.info("Initializing Modern Crypto Trader...")

    strategy = MLStrategy()

    # Check if model exists, if not, train it
    if not strategy.model.load():
        logging.info("Model not found. Fetching historical data for training...")
        fetcher = DataFetcher()
        hist_df = fetcher.fetch_historical_ohlcv(Config.SYMBOL, Config.INTERVAL, 2023)
        strategy.train(hist_df)

    executor = BinanceExecutor(mock=Config.MOCK_MODE)

    def candle_callback(candle):
        price = candle[4] # Close
        logging.info(f"New Candle: {candle[0]} - Price: {price}")

        executor.check_risk_management(price)
        signal_action = strategy.on_candle(candle)

        if signal_action != "HOLD":
            logging.info(f"Signal Generated: {signal_action}")
            executor.execute_signal(signal_action, price)

    streamer = DataStreamer(
        symbol=Config.SYMBOL,
        interval=Config.INTERVAL,
        callback=candle_callback
    )

    def signal_handler(sig, frame):
        logging.info("Shutting down bot...")
        streamer.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        streamer.start()
    except Exception as e:
        logging.error(f"Critical Error: {e}")
        streamer.stop()

if __name__ == "__main__":
    main()
