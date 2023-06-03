import ccxt
import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LinearRegression
from binance.client import Client
from binance.enums import *
from tensorflow.keras.models import load_model
from lstm_model import get_latest_data_and_predict
from lstm_model import create_lstm_model
from lstm_model import retrain_model
from preprocessing import add_features
from tradeLogic import execute_trade

def trader():
    # Constants
    TRADE_FRACTION = 0.05  # Trade 5% of the balance
    TIME_BETWEEN_TRADES = 5  # 5 seconds between trades
    MAX_ACTIVE_TRADES = 5 # Maximum number of active trades
    STOP_LOSS_PERCENTAGE = 0.99 # Stop loss at 1% loss
    window = 3  # Adjusted window size for hourly data
    slope_threshold = 0.01  # Adjusted slope threshold for increased sensitivity
    file_name = f"BTCUSDT_5m_filtered_data_with_features.csv" # File name of the preprocessed data
    mock_balance = {"USDT": 1000, "BTC": 0} # Initialize the mock balance
    active_trades = [] # Initialize the active trades array
    rising_trend = False # Initialize the rising trend flag
    rising_trend_array = [] # Initialize the rising trend array
    dropping_trend = False # Initialize the dropping trend flag
    dropping_trend_array = [] # Initialize the dropping trend array
    total_trades_made = 0 # Initialize the total trades made counter
    historical_prices = [] # Initialize the historical prices array
    predicted_price_array = [] # Initialize the predicted price array
    retrain_counter = 0 # Initialize the retrain counter
    iteration = 1 # Initialize the iteration counter
    retrain_interval = 10  # Retrain the model every 10 iterations
    last_retrain_time = time.time() # Initialize the last retrain time
    live_data_file_path = "live_data.csv" # Path to the live data file
    live_data_file = open(live_data_file_path, "a")  # Open the live data file
    live_data_file.write("Open time,Open,High,Low,Close,Volume\n") # Write the header to the live data file
    symbol = 'BTC/USDT' # Define the symbol to be used
    interval = '1m' # Define the interval to be used
    lstm_model_path = f"BTCUSDT_5m_lstm_model.h5" # Path to the LSTM model
    model_lstm = load_model(lstm_model_path) # Load the LSTM model
    api_key = '5LP8ofECmS4alejitd2fwnYriEF96dI7BrTliB7oM5popffVskoi8hOqXalIQjfQ' # Define the API key
    api_secret = 'VIOL4sEXWMChq4ecG4OxfZXmBmDibyNx3uIpQiKBWlauhFbfpAnygiN9utOgloC4' # Define the API secret
    lookback = 30 # Define the lookback period
    data = pd.read_csv(f"BTCUSDT_5m_filtered_data_with_features.csv") # Load the preprocessed data with features
    train_split_ratio = 0.8 # Define the train split ratio
    train_size = int(len(data) * train_split_ratio) # Calculate the train size
    train_data = data[:train_size] # Define the train data
    validation_data = data[train_size:] # Define the validation data
    _, scaler = create_lstm_model(data, lookback, epochs=0, batch_size=1) # Create the scaler
    log_file = open("trade_log.csv", "a") # Open the log file
    log_file.write("Timestamp,Current_Price,Predicted_Price\n") # Write the header to the log file
    last_logged_timestamp = -1 # Initialize the last logged timestamp

    # Initialize the Binance API
    exchange = ccxt.binance({
        'apiKey': api_key,
        'secret': api_secret,
    })

    def calculate_slope(prices, window=300):
        model = LinearRegression()
        x = np.arange(window).reshape(-1, 1)
        y = prices[-window:].reshape(-1, 1)
        model.fit(x, y)
        return model.coef_[0][0]



    # Main loop for the trading bot
    while True:
        try:
            iteration += 1 # Increment the iteration counter

            # Get the current balance of your account
            # balance = exchange.fetch_balance()

            # Get the latest data and make a prediction
            predicted_change, current_price = get_latest_data_and_predict(model_lstm, exchange, symbol, interval, lookback, scaler)
            predicted_price = current_price * (1 + predicted_change)

            # Print the current price and predicted price comparison
            print(f"Current Price: {current_price}, Predicted Price: {predicted_price}")
            historical_prices.append(current_price)
            
            # generate the predicted price array and pop the first element if the array is greater than 10 and print the array
            predicted_price_array.append(predicted_price)
            if len(predicted_price_array) > 12:
                predicted_price_array.pop(0)
            print(f"Predicted average is {sum(predicted_price_array)/len(predicted_price_array)}")

            # Calculate the slope of the regression line for the given window
            slope = calculate_slope(np.array(historical_prices), window)

            # generate the rising trend array and pop the first element if the array is greater than 2 and print the array
            rising_trend_array.append(current_price)
            dropping_trend_array.append(current_price)
            print(f"Trend array values: {rising_trend_array[0]}, {rising_trend_array[1]}, {rising_trend_array[2]}")
            # Check for rising trend
            print(f"{slope} > {slope_threshold}")
            if slope > slope_threshold:
                rising_trend = True
            else:
                rising_trend = False
            # Check for dropping trend
            if slope < -slope_threshold:
                dropping_trend = True
            else:
                dropping_trend = False

            # Log the data to the file
            log_file.write(f"{int(time.time())},{current_price},{predicted_price}\n")
            log_file.flush()

            # Log the latest data to the live data file for retraining
            live_data = exchange.fetch_ohlcv(symbol, interval)
            latest_data = live_data[-1]

            current_timestamp = latest_data[0] // 1000  # Convert to seconds
            if current_timestamp > last_logged_timestamp + 60 * 5 :  # Check if at least one minute has passed
                live_data_file.write(f"{latest_data[0]},{latest_data[1]},{latest_data[2]},{latest_data[3]},{latest_data[4]},{latest_data[5]}\n")
                live_data_file.flush()
                last_logged_timestamp = current_timestamp  # Update the last logged timestamp
                print("Logged live prediction data")

            # Retrain the model every 2 hours using the logged live data
            if iteration % 5 == 0:

                live_data_df = pd.read_csv("live_data.csv")
                live_data_df = add_features(live_data_df)

                if live_data_df.empty:
                    print("DataFrame is empty after adding features. Skipping retraining.")
                else:
                    live_data_normalized = scaler.transform(live_data_df)
                    retrain_model(model_lstm, live_data_normalized, lookback, batch_size=1, epochs=10)
                    print("Retrained the model")
                    retrain_counter += 1

            print("pre execute trade")
            # Execute trade based on the predicted price
            execute_trade(mock_balance, current_price, predicted_price, TRADE_FRACTION, active_trades, total_trades_made, rising_trend, dropping_trend, predicted_price_array, retrain_interval, MAX_ACTIVE_TRADES, STOP_LOSS_PERCENTAGE)
            print("post execute trade")

            # Wait for the next iteration
            time.sleep(TIME_BETWEEN_TRADES)  # 5 seconds

        except Exception as e:
            print(f"Error: {str(e)}")
            time.sleep(7)  # Sleep for 5 seconds before trying again

    # Close the log
    log_file.close()

trader()
