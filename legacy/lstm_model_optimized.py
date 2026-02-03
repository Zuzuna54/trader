import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back, -1])
    return np.array(X), np.array(Y)

def create_lstm_model(data, look_back=1, epochs=100, batch_size=1):
    # Normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(data)
    
    # Split the dataset into train and test sets
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    
    # Reshape the dataset for the LSTM model
    X_train, Y_train = create_dataset(train, look_back)
    X_test, Y_test = create_dataset(test, look_back)
    
    # Create the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    # Train the LSTM model
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=2)
    
    return model, scaler


data = pd.read_csv("BTCUSDT_5m_filtered_data_with_features.csv")
lookback = 6
epochs = 100
batch_size = 1
lstm_model, scaler = create_lstm_model(data, lookback, epochs, batch_size)
lstm_model.save("BTCUSDT_5m_lstm_model.h5")