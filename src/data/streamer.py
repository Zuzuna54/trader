import ccxt
import time
import logging
from src.config import Config

class DataStreamer:
    def __init__(self, symbol, interval, callback, exchange_id='binance'):
        self.symbol = symbol
        self.interval = interval
        self.callback = callback
        self.exchange = getattr(ccxt, exchange_id)({
            'apiKey': Config.API_KEY,
            'secret': Config.API_SECRET,
        })
        self.is_running = False

    def start(self):
        logging.info(f"Starting data streamer for {self.symbol} ({self.interval})")
        self.is_running = True
        last_timestamp = None

        while self.is_running:
            try:
                ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.interval, limit=2)
                if ohlcv:
                    current_candle = ohlcv[-1]
                    timestamp = current_candle[0]

                    if timestamp != last_timestamp:
                        last_timestamp = timestamp
                        self.callback(current_candle)

                time.sleep(5) # Poll every 5 seconds
            except Exception as e:
                logging.error(f"Streamer error: {e}")
                time.sleep(10)

    def stop(self):
        self.is_running = False
