import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional

def fetch_stock_data(symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch stock data from Yahoo Finance API
    """
    stock = yf.Ticker(symbol)
    df = stock.history(start=start_date, end=end_date)
    return df

def prepare_data(df: pd.DataFrame, sequence_length: int = 60, train_split: float = 0.8) -> Tuple[np.ndarray, ...]:
    """
    Prepare data for model training
    """
    # Select features
    data = df[['Close', 'Volume', 'High', 'Low']].values
    
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Create sequences
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:(i + sequence_length)])
        y.append(scaled_data[i + sequence_length, 0])  # Predicting Close price
    
    X = np.array(X)
    y = np.array(y)
    
    # Split into train and test sets
    train_size = int(len(X) * train_split)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, X_test, y_train, y_test, scaler

def inverse_transform_predictions(predictions: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """
    Convert scaled predictions back to original scale
    """
    # Create a dummy array with same shape as training data
    dummy = np.zeros((len(predictions), 4))
    dummy[:, 0] = predictions  # Close price predictions
    return scaler.inverse_transform(dummy)[:, 0] 