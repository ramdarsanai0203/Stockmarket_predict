# Indian Stock Market Prediction for Multiple Stocks using LSTM in Python
# This script predicts the closing prices for multiple Indian stocks (e.g., Reliance, Tata Motors, NIFTY 50) using Long Short-Term Memory (LSTM) networks.
# It fetches historical data using yfinance, preprocesses it, trains separate LSTM models for each stock, and makes predictions.
# Requirements: Install necessary libraries if not already installed.
# pip install yfinance pandas numpy tensorflow scikit-learn matplotlib

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta

# Step 1: Define parameters
tickers = ['RELIANCE.NS', 'TATAMOTORS.NS', '^NSEI']  # List of tickers: Change or add more Indian stocks/indices
start_date = '2015-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')  # Current date
time_step = 60  # Look back 60 days
future_days = 30  # Predict next 30 days
train_ratio = 0.8  # 80% for training

# Function to fetch and preprocess data for a single ticker
def get_stock_data(ticker):
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        print(f"No data found for {ticker}")
        return None
    df = df[['Close']]
    df.dropna(inplace=True)
    return df

# Function to create dataset with time steps
def create_dataset(scaled_data, time_step):
    X, y = [], []
    for i in range(len(scaled_data) - time_step - 1):
        X.append(scaled_data[i:(i + time_step), 0])
        y.append(scaled_data[i + time_step, 0])
    return np.array(X), np.array(y)

# Function to build and train LSTM model
def build_and_train_model(X_train, y_train, X_test, y_test):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=0)
    return model

# Function to make future predictions
def predict_future(model, scaled_data, scaler, time_step, future_days):
    last_60_days = scaled_data[-time_step:]
    future_predictions = []
    current_input = last_60_days.reshape(1, time_step, 1)
    for _ in range(future_days):
        future_pred = model.predict(current_input, verbose=0)
        future_predictions.append(future_pred[0, 0])
        current_input = np.append(current_input[:, 1:, :], future_pred.reshape(1, 1, 1), axis=1)
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_predictions

# Main loop for each ticker
for ticker in tickers:
    print(f"\nProcessing {ticker}...")
    
    # Fetch data
    df = get_stock_data(ticker)
    if df is None:
        continue
    
    # Visualize closing price history
    plt.figure(figsize=(16,8))
    plt.title(f'Closing Price History of {ticker}')
    plt.plot(df['Close'])
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price INR (₹)', fontsize=18)
    plt.show()
    
    # Preprocess data
    data = df.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Split into train and test
    train_len = int(len(scaled_data) * train_ratio)
    train_data = scaled_data[0:train_len, :]
    test_data = scaled_data[train_len:, :]
    
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    
    if len(X_train) == 0 or len(X_test) == 0:
        print(f"Insufficient data for {ticker} to train model.")
        continue
    
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Build and train model
    model = build_and_train_model(X_train, y_train, X_test, y_test)
    
    # Predict on test data
    predicted_prices = model.predict(X_test, verbose=0)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    
    # Visualize actual vs predicted on test set
    # plt.figure(figsize=(16,8))
    # plt.title(f'Stock Price Prediction for {ticker}')
    # plt.plot(data[train_len + time_step + 1:], label='Actual Price (Test Set)')
    # plt.plot(predicted_prices, label='Predicted Price (Test Set)')
    # plt.xlabel('Time', fontsize=18)
    # plt.ylabel('Price INR (₹)', fontsize=18)
    # plt.legend()
    # plt.show()
    
    # Future predictions
    future_predictions = predict_future(model, scaled_data, scaler, time_step, future_days)
    
    # Plot future predictions
    future_dates = pd.date_range(df.index[-1] + timedelta(days=1), periods=future_days)
    # plt.figure(figsize=(16,8))
    # plt.title(f'Future Stock Price Prediction for {ticker} (Next {future_days} Days)')
    # plt.plot(future_dates, future_predictions, label='Predicted Future Price')
    # plt.xlabel('Date', fontsize=18)
    # plt.ylabel('Price INR (₹)', fontsize=18)
    # plt.legend()
    # plt.show()
    
    # Print predicted prices for the next 30 days
    print(f"Predicted prices for {ticker} for the next {future_days} days:")
    for i in range(future_days):
        print(f"Day {i+1} ({future_dates[i].date()}): ₹{future_predictions[i][0]:.2f}")