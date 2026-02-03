import ccxt
import pandas as pd
import ta
import talib
import time
from datetime import datetime
import logging


# Add features to the dataframe using the ta library
def add_features(df):

    # Add all ta features filling nans values
    df['Open time'] = df['Open time'].astype(int)  # Convert 'Open time' to integer
    df['Close time'] = df['Open time'] + 5 * 60 * 1000
    df['EMA_9'] = ta.trend.EMAIndicator(df['Close'], window=9).ema_indicator()
    df['EMA_21'] = ta.trend.EMAIndicator(df['Close'], window=21).ema_indicator()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    df['BB_high'], df['BB_mid'], df['BB_low'] = ta.volatility.BollingerBands(df['Close']).bollinger_hband(), ta.volatility.BollingerBands(df['Close']).bollinger_mavg(), ta.volatility.BollingerBands(df['Close']).bollinger_lband()
    df['ROC'] = talib.ROC(df['Close'], timeperiod=10)
    df['Stoch_k'], df['Stoch_d'] = talib.STOCH(df['High'], df['Low'], df['Close'], fastk_period=14, slowk_period=3, slowd_period=3)

    # Add the number of trades as a feature
    if 'Taker buy base asset volume' in df.columns and 'Taker buy quote asset volume' in df.columns:
        df['Number of trades'] = df['Taker buy base asset volume'] + df['Taker buy quote asset volume']
    else:
        df['Number of trades'] = 0

    # Drop the NaN values
    df.dropna(inplace=True)

    return df


# 
def fetch_all_historical_data(exchange, symbol, interval, start_year):

    all_data = []
    timeframe = exchange.parse_timeframe(interval)
    max_limit = 1000  # Binance's maximum limit per request
    now = exchange.milliseconds()

    # Start from the specified year.
    start_date = datetime(start_year, 1, 1)
    start_timestamp = int(start_date.timestamp() * 1000)
    
    while True:
        try:
            if now - start_timestamp <= 0:
                break
            data = exchange.fetch_ohlcv(symbol, interval, limit=max_limit, since=now - (timeframe * max_limit * 1000))
            all_data = data + all_data
            now = data[0][0]
            start_date = datetime.fromtimestamp(now / 1000).strftime('%d/%m/%Y')
            end_date = datetime.fromtimestamp(data[-1][0] / 1000).strftime('%d/%m/%Y')
            print(f"Fetched {len(data)} data points from {start_date} to {end_date}")
            time.sleep(exchange.rateLimit / 1000)  # Respect the rate limit
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(exchange.rateLimit / 1000)
            continue

    return all_data


# Preprocess the data and save it to a CSV file
def preprocess_data(symbol, interval, output_file_name, start_year):
    # Initialize the Binance API
    exchange = ccxt.binance()

    # Fetch historical data
    
    try:
        historical_data = fetch_all_historical_data(exchange, symbol, interval, start_year)
    except ccxt.NetworkError as e:
        logging.error(f"Network error: {e}")
        return
    except ccxt.ExchangeError as e:
        logging.error(f"Exchange error: {e}")
        return

    # Check the length of a single data row
    data_length = len(historical_data[0])

    # Adjust the columns list based on the data_length
    if data_length == 6:
        columns = ["Open time", "Open", "High", "Low", "Close", "Volume"]
    else:
        columns = ["Open time", "Open", "High", "Low", "Close", "Volume", "Close time", "Quote asset volume", "Number of trades", "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"]

    # Convert the historical data into a DataFrame
    df = pd.DataFrame(historical_data, columns=columns)

    # Apply the feature extraction
    df = add_features(df)

    # Save the preprocessed data to a CSV file
    try:
        df.to_csv(output_file_name, index=False)
        logging.info(f"Data saved to {output_file_name}")
    except Exception as e:
        logging.error(f"Error while saving data to file: {e}")


if __name__ == "__main__":
    symbol = 'BTC/USDT'
    interval = '5m'
    start_year = 2018  # Set start year as 2018
    file_name = f"BTCUSDT_{interval}_filtered_data_with_features.csv"
    preprocess_data(symbol, interval, file_name, start_year)