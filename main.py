
import pandas as pd

from preprocessing import preprocess_data
from lstm_model import create_lstm_model
from trader import trader

symbol = 'BTC/USDT'
interval = '5m'
start_year = 2020  # Set start year as 2018
file_name = f"BTCUSDT_{interval}_filtered_data_with_features.csv"
preprocess_data(symbol, interval, file_name, start_year)


data = pd.read_csv("BTCUSDT_5m_filtered_data_with_features.csv")
lookback = 30
epochs = 50
batch_size = 64
lstm_model, scaler = create_lstm_model(data, lookback, epochs, batch_size)
lstm_model.save("BTCUSDT_5m_lstm_model.h5")