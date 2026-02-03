import ccxt
import pandas as pd
import time
from datetime import datetime
import logging

class DataFetcher:
    def __init__(self, exchange_id='binance'):
        self.exchange = getattr(ccxt, exchange_id)()

    def fetch_historical_ohlcv(self, symbol, interval, start_year):
        all_data = []
        timeframe = self.exchange.parse_timeframe(interval)
        max_limit = 1000

        # Start from the specified year
        start_date = datetime(start_year, 1, 1)
        since = int(start_date.timestamp() * 1000)

        logging.info(f"Fetching historical data for {symbol} starting from {start_year}...")

        while True:
            try:
                data = self.exchange.fetch_ohlcv(symbol, interval, limit=max_limit, since=since)
                if not data:
                    break

                all_data += data
                since = data[-1][0] + (timeframe * 1000)

                last_date = datetime.fromtimestamp(data[-1][0] / 1000)
                logging.info(f"Fetched up to {last_date}")

                if data[-1][0] >= self.exchange.milliseconds() - (timeframe * 1000):
                    break

                time.sleep(self.exchange.rateLimit / 1000)
            except Exception as e:
                logging.error(f"Error fetching data: {e}")
                time.sleep(1)
                continue

        df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
