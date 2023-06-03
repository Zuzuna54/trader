import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from preprocessing import add_features
from keras.layers import Dropout, Bidirectional


# Create the input and output datasets
def create_dataset(dataset, lookback):
    X, y = [], []
    for i in range(lookback, len(dataset)):
        X.append(dataset[i - lookback:i, :-2])
        y.append(dataset[i, -1])
    return np.array(X), np.array(y)


# Create the LSTM model with params
def create_lstm_model(input_data, lookback, epochs=50, batch_size=64):
    # Normalize the data
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(input_data)

    # Create the input and output datasets
    X, y = create_dataset(data_normalized, lookback)

    # Create the LSTM model
    model = Sequential()
    model.add(Bidirectional(LSTM(100, return_sequences=True), input_shape=(lookback, X.shape[2])))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(100)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mean_squared_error",)

    # Train the LSTM model
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)

    return model, scaler

# Function to get the latest data and make a prediction
def get_latest_data_and_predict(model, exchange, symbol, interval, lookback, scaler):
    # Fetch the latest historical data
    historical_data = exchange.fetch_ohlcv(symbol, interval)
    # Convert the historical data into a DataFrame
    df = pd.DataFrame(historical_data, columns=["Open time", "Open", "High", "Low", "Close", "Volume"])
    # Apply the same preprocessing steps as before
    df = add_features(df)
    # Normalize the data
    df_normalized = scaler.transform(df)
    # Create the input dataset
    X_latest, _ = create_dataset(df_normalized, lookback)
    # Make a prediction using the LSTM model
    y_pred = model.predict(X_latest[-1].reshape(1, lookback, -1))

    return y_pred[0][0], df.iloc[-1]['Close']


# Function to retrain the LSTM model
def retrain_model(model, data, lookback, batch_size, epochs):
    X, y = create_dataset(data, lookback)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)


# Example usage
data = pd.read_csv("BTCUSDT_5m_filtered_data_with_features.csv")
lookback = 30
epochs = 50
batch_size = 64
lstm_model, scaler = create_lstm_model(data, lookback, epochs, batch_size)
lstm_model.save("BTCUSDT_5m_lstm_model.h5")