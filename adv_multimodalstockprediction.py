# Enhanced Multimodal Stock Prediction Script with Robust NaN Handling
# Integrates prices, financials, technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, Volatility),
# news sentiment, Transformer/LSTM model, and Monte Carlo simulations.
# Requirements: pip install yfinance pandas numpy tensorflow scikit-learn matplotlib openpyxl textblob
# Optional: pip install tweepy (for X sentiment)

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN message
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, MultiHeadAttention, LayerNormalization, Conv1D, Dropout, GlobalAveragePooling1D, Dense, LSTM
from tensorflow.keras.optimizers import Adam
from datetime import datetime, timedelta
import time

# Check for TextBlob
try:
    from textblob import TextBlob
except ImportError:
    print("TextBlob not installed. Sentiment analysis skipped. Install with: pip install textblob")
    TextBlob = None

# Parameters
tickers = ['SBIN.NS','TORNTPOWER.NS','NYKAA.NS','NILAINFRA.NS','MMP.NS','BSE.NS']
start_date = '2020-01-01'
end_date = '2025-09-15'
time_step = 60
future_days = 30
train_ratio = 0.8
excel_file = 'enhanced_multimodal_predictions.xlsx'
monte_carlo_runs = 100

# Function to compute technical indicators
def compute_technical_indicators(df):
    try:
        print(f"Computing technical indicators for {len(df)} rows")
        df = df.copy()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        rolling_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['SMA_20'] + (rolling_std * 2)
        df['BB_Lower'] = df['SMA_20'] - (rolling_std * 2)
        
        df['Returns'] = df['Close'].pct_change(fill_method=None)
        df['Volatility_20'] = df['Returns'].rolling(window=20).std()
        
        # Fill NaN with forward-fill and drop remaining
        df = df.ffill()
        df_clean = df.dropna()
        print(f"After indicators, {len(df_clean)} rows remain")
        return df_clean
    except Exception as e:
        print(f"Error computing technical indicators: {e}")
        return df[['Close']].ffill()

# Function to get sentiment
def get_sentiment(ticker):
    sentiment = 0.0
    if TextBlob is None:
        return sentiment
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if news:
            sentiments = [TextBlob(article.get('title', '')).sentiment.polarity for article in news[:5]]
            sentiment = np.mean(sentiments) if sentiments else 0.0
        # Placeholder for X sentiment (requires Tweepy and API key)
        return sentiment
    except Exception as e:
        print(f"Error fetching sentiment for {ticker}: {e}")
        return 0.0

# Function to fetch data
def fetch_data(ticker):
    try:
        time.sleep(1)
        price_df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)[['Close']]
        if price_df.empty:
            print(f"No price data for {ticker}")
            return None
        
        price_df = price_df.reset_index()
        if isinstance(price_df.columns, pd.MultiIndex):
            price_df.columns = price_df.columns.get_level_values(0)
        price_df = price_df[['Date', 'Close']]
        price_df['Date'] = pd.to_datetime(price_df['Date'])
        price_df = price_df.set_index('Date')
        print(f"Fetched {len(price_df)} rows for {ticker}")
        
        price_df = compute_technical_indicators(price_df)
        if price_df.empty or len(price_df) < time_step + 1:
            print(f"Insufficient data after indicators for {ticker}: {len(price_df)} rows")
            return price_df
        
        if ticker.endswith('.NS') or ticker.endswith('.BO'):
            try:
                stock = yf.Ticker(ticker)
                income = stock.quarterly_financials
                balance = stock.quarterly_balance_sheet
                if not income.empty and not balance.empty:
                    income = income.T
                    balance = balance.T
                    financial_df = pd.DataFrame(index=income.index)
                    financial_df['Net Income'] = income.get('Net Income', np.nan)
                    financial_df['Revenue'] = income.get('Total Revenue', np.nan)
                    financial_df['Shares Outstanding'] = balance.get('Common Stock Shares Outstanding', np.nan)
                    financial_df['Shareholders Equity'] = balance.get('Total Stockholder Equity', np.nan)
                    financial_df['Total Liabilities'] = balance.get('Total Liabilities', np.nan)
                    
                    financial_df['EPS'] = financial_df['Net Income'] / financial_df['Shares Outstanding']
                    financial_df['ROE'] = financial_df['Net Income'] / financial_df['Shareholders Equity']
                    financial_df['Debt_to_Equity'] = financial_df['Total Liabilities'] / financial_df['Shareholders Equity']
                    financial_df['Revenue_Growth'] = financial_df['Revenue'].pct_change(fill_method=None)
                    
                    financial_df = financial_df.ffill()
                    if not financial_df.empty:
                        daily_index = pd.date_range(start=max(financial_df.index.min(), price_df.index.min()), end=price_df.index.max(), freq='D')
                        financial_df = financial_df.reindex(daily_index, method='ffill')
                        price_df = price_df.merge(financial_df, left_index=True, right_index=True, how='left')
                        price_df = price_df.ffill()
                        print(f"After financials, {len(price_df)} rows remain")
                    else:
                        print(f"No valid financial data for {ticker}, using price and indicators")
                else:
                    print(f"No financial data for {ticker}, using price and indicators")
            except Exception as e:
                print(f"Error fetching financial data for {ticker}: {e}, using price and indicators")
        
        if price_df.empty or len(price_df) < time_step + 1:
            print(f"Insufficient data after processing for {ticker}: {len(price_df)} rows")
            return price_df
        
        return price_df
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

# Transformer encoder
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    try:
        x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
        x = Dropout(dropout)(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs
        x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
        x = Dropout(dropout)(x)
        x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        return x + res
    except Exception as e:
        print(f"Error in transformer encoder: {e}")
        return inputs

# Fallback LSTM model
def build_lstm_model(num_features):
    try:
        model = Sequential([
            Input(shape=(time_step, num_features)),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(units=1)
        ])
        model.compile(optimizer=Adam(clipnorm=1.0), loss='mean_squared_error')
        return model
    except Exception as e:
        print(f"Error building LSTM model: {e}")
        return None

# Function to build and train model
def build_and_train_model(X_train, y_train, X_test, y_test, num_features):
    try:
        print(f"Training Transformer model with {num_features} features")
        inputs = Input(shape=(time_step, num_features))
        x = inputs
        for _ in range(2):
            x = transformer_encoder(x, head_size=64, num_heads=2, ff_dim=4, dropout=0.2)
        x = GlobalAveragePooling1D(data_format="channels_last")(x)
        x = Dense(64, activation="relu")(x)
        x = Dropout(0.2)(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(clipnorm=1.0), loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)
        return model
    except Exception as e:
        print(f"Transformer failed: {e}, falling back to LSTM")
        model = build_lstm_model(num_features)
        if model:
            model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)
        return model

# Function to create dataset
def create_dataset(data, time_step):
    try:
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), :])
            y.append(data[i + time_step, 0])
        X, y = np.array(X), np.array(y)
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            print("NaN detected in dataset")
            return np.array([]), np.array([])
        print(f"Created dataset: X shape {X.shape}, y shape {y.shape}")
        return X, y
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return np.array([]), np.array([])

# Function to predict future prices with Monte Carlo
def predict_future_with_monte_carlo(model, scaled_data, scaler, time_step, future_days, num_features, sentiment, monte_carlo_runs):
    try:
        all_predictions = []
        for _ in range(monte_carlo_runs):
            future_predictions = []
            current_input = scaled_data[-time_step:].copy().reshape(1, time_step, num_features)
            for _ in range(future_days):
                noise = np.random.normal(0, 0.01, current_input.shape)
                noisy_input = current_input + noise
                if np.any(np.isnan(noisy_input)):
                    print("NaN detected in Monte Carlo input")
                    return np.array([]), np.array([])
                future_pred = model.predict(noisy_input, verbose=0)[0, 0]
                if np.isnan(future_pred):
                    print("NaN detected in model prediction")
                    return np.array([]), np.array([])
                future_predictions.append(future_pred)
                new_row = np.append(future_pred, np.append(scaled_data[-1, 1:-1], sentiment)).reshape(1, 1, num_features) if 'Sentiment' in features else np.append(future_pred, scaled_data[-1, 1:]).reshape(1, 1, num_features)
                current_input = np.append(current_input[:, 1:, :], new_row, axis=1)
            all_predictions.append(future_predictions)
        
        all_predictions = np.array(all_predictions)
        if np.any(np.isnan(all_predictions)):
            print("NaN detected in Monte Carlo predictions")
            return np.array([]), np.array([])
        mean_predictions = np.mean(all_predictions, axis=0)
        std_predictions = np.std(all_predictions, axis=0)
        mean_predictions = scaler.inverse_transform(np.concatenate((mean_predictions.reshape(-1, 1), np.zeros((future_days, num_features - 1))), axis=1))[:, 0]
        if np.any(np.isnan(mean_predictions)):
            print("NaN detected in inverse-transformed predictions")
            return np.array([]), np.array([])
        return mean_predictions, std_predictions
    except Exception as e:
        print(f"Error in Monte Carlo predictions: {e}")
        return np.array([]), np.array([])

# Main loop
all_future_predictions = {}
all_uncertainties = {}
future_dates = None
features = None
for ticker in tickers:
    print(f"\nProcessing {ticker}...")
    
    df = fetch_data(ticker)
    if df is None or df.empty:
        print(f"Skipping {ticker} due to empty data")
        continue
    
    sentiment = get_sentiment(ticker)
    if TextBlob is not None:
        df['Sentiment'] = sentiment
    
    available_financial = [col for col in ['EPS', 'ROE', 'Debt_to_Equity', 'Revenue_Growth'] if col in df.columns]
    technical_features = ['SMA_20', 'EMA_12', 'RSI_14', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'Volatility_20']
    features = ['Close'] + [f for f in technical_features if f in df.columns] + available_financial
    if TextBlob is not None:
        features += ['Sentiment']
    
    try:
        # Remove constant features to avoid scaling issues
        data = df[features].values
        valid_features = []
        for i, feature in enumerate(features):
            if np.std(data[:, i]) > 1e-6:  # Avoid scaling constant columns
                valid_features.append(feature)
            else:
                print(f"Skipping constant feature {feature} for {ticker}")
        if not valid_features:
            print(f"No valid features for {ticker}, using Close only")
            features = ['Close']
            data = df[['Close']].values
        else:
            features = valid_features
            data = df[features].values
        print(f"Selected features: {features}, data shape: {data.shape}")
    except KeyError as e:
        print(f"Missing features for {ticker}: {e}, using Close only")
        features = ['Close']
        data = df[['Close']].values
    
    if len(data) < time_step + 1:
        print(f"Insufficient data for {ticker}: {len(data)} rows, need {time_step + 1}")
        continue
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    try:
        if np.any(np.isnan(data)):
            print(f"NaN detected in data for {ticker}, filling with mean")
            data = np.nan_to_num(data, nan=np.nanmean(data, axis=0))
        scaled_data = scaler.fit_transform(data)
        if np.any(np.isnan(scaled_data)):
            print(f"NaN detected in scaled data for {ticker}")
            continue
    except Exception as e:
        print(f"Error scaling data for {ticker}: {e}")
        continue
    
    train_len = int(len(scaled_data) * train_ratio)
    train_data = scaled_data[0:train_len, :]
    test_data = scaled_data[train_len:, :]
    
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    
    if len(X_train) == 0 or len(X_test) == 0:
        print(f"Insufficient dataset for {ticker}: X_train {len(X_train)}, X_test {len(X_test)}")
        continue
    
    num_features = X_train.shape[2]
    
    model = build_and_train_model(X_train, y_train, X_test, y_test, num_features)
    if model is None:
        print(f"Model training failed for {ticker}")
        continue
    
    try:
        predicted_prices = model.predict(X_test, verbose=0)
        predicted_prices = scaler.inverse_transform(np.concatenate((predicted_prices, np.zeros((len(predicted_prices), num_features - 1))), axis=1))[:, 0]
        if np.any(np.isnan(predicted_prices)):
            print(f"NaN detected in test predictions for {ticker}")
            continue
    except Exception as e:
        print(f"Error predicting test prices for {ticker}: {e}")
        continue
    
    test_dates = df.index[train_len + time_step + 1:]
    test_actual = data[train_len + time_step + 1:, 0]
    plt.figure(figsize=(16,8))
    plt.title(f'Stock Price Prediction for {ticker} (Enhanced Model)')
    plt.plot(test_dates, test_actual, label='Actual Price')
    plt.plot(test_dates, predicted_prices, label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price INR (₹)')
    plt.legend()
    plt.savefig(f'prediction_{ticker}.png')
    plt.close()
    
    scaled_sentiment = scaler.transform(np.concatenate((np.zeros((1, num_features - 1)), [[sentiment]]), axis=1))[0, -1] if 'Sentiment' in features else 0
    mean_predictions, std_predictions = predict_future_with_monte_carlo(model, scaled_data, scaler, time_step, future_days, num_features, scaled_sentiment, monte_carlo_runs)
    if len(mean_predictions) == 0:
        print(f"Future predictions failed for {ticker}")
        continue
    
    all_future_predictions[ticker] = mean_predictions
    all_uncertainties[ticker] = std_predictions
    
    if future_dates is None:
        future_dates = pd.date_range(df.index[-1] + timedelta(days=1), periods=future_days)
    
    plt.figure(figsize=(16,8))
    plt.title(f'Future Prediction for {ticker} (Next {future_days} Days) with Uncertainty')
    plt.plot(future_dates, mean_predictions, label='Mean Predicted Price')
    plt.fill_between(future_dates, mean_predictions - std_predictions, mean_predictions + std_predictions, color='b', alpha=0.2, label='Uncertainty (1 std)')
    plt.xlabel('Date')
    plt.ylabel('Price INR (₹)')
    plt.legend()
    plt.savefig(f'future_prediction_{ticker}.png')
    plt.close()
    
    print(f"Predicted prices for {ticker} (Mean ± Std):")
    for i in range(future_days):
        print(f"Day {i+1} ({future_dates[i].date()}): ₹{mean_predictions[i]:.2f} ± {std_predictions[i]:.2f}")

# Save to Excel
if all_future_predictions:
    future_df = pd.DataFrame({'Date': future_dates})
    for ticker in all_future_predictions:
        future_df[f'{ticker} Mean Predicted Price (INR)'] = all_future_predictions[ticker]
        future_df[f'{ticker} Uncertainty Std (INR)'] = all_uncertainties[ticker]
    try:
        future_df.to_excel(excel_file, sheet_name='Future_Predictions', index=False)
        print(f"\nPredictions saved to {excel_file}")
    except Exception as e:
        print(f"Error saving to Excel: {e}")
else:
    print("No predictions to save due to data issues")