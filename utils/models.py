import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from typing import Tuple, Optional
from .sentiment_analysis import NewsAnalyzer
import pandas as pd
from datetime import datetime

class LSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class StockPredictor:
    def __init__(self, model_type: str = 'lstm', input_dim: int = 5,  # Updated to 5 to include sentiment
                 hidden_dim: int = 64, num_layers: int = 2):
        self.model_type = model_type
        self.news_analyzer = NewsAnalyzer()
        if model_type == 'lstm':
            self.model = LSTM(input_dim, hidden_dim, num_layers, output_dim=1)
        else:
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            
    def prepare_features_with_sentiment(self, X: np.ndarray, symbol: str) -> np.ndarray:
        """Add sentiment features to the input data"""
        # Get sentiment scores for the period
        articles = self.news_analyzer.fetch_financial_news(symbol)
        sentiment_scores = []
        
        # Calculate daily sentiment scores
        dates = pd.date_range(end=datetime.now(), periods=len(X))
        for date in dates:
            day_articles = [a for a in articles if a['date'].date() == date.date()]
            sentiment_scores.append(self.news_analyzer.get_aggregated_sentiment(day_articles))
        
        # Add sentiment as a new feature
        sentiment_array = np.array(sentiment_scores).reshape(-1, 1)
        return np.concatenate([X, sentiment_array], axis=2)
            
    def train(self, X_train: np.ndarray, y_train: np.ndarray, symbol: str,
             epochs: int = 100, batch_size: int = 32) -> Optional[list]:
        # Add sentiment features
        X_train_with_sentiment = self.prepare_features_with_sentiment(X_train, symbol)
        
        if self.model_type == 'lstm':
            return self._train_lstm(X_train_with_sentiment, y_train, epochs, batch_size)
        else:
            return self._train_rf(X_train_with_sentiment, y_train)
    
    def _train_lstm(self, X_train: np.ndarray, y_train: np.ndarray, 
                    epochs: int, batch_size: int) -> list:
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters())
        
        losses = []
        for epoch in range(epochs):
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                
            losses.append(loss.item())
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
                
        return losses
    
    def _train_rf(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        # Reshape data for Random Forest
        X_train_2d = X_train.reshape(X_train.shape[0], -1)
        self.model.fit(X_train_2d, y_train)
        return None
    
    def predict(self, X: np.ndarray, symbol: str) -> np.ndarray:
        # Add sentiment features
        X_with_sentiment = self.prepare_features_with_sentiment(X, symbol)
        
        if self.model_type == 'lstm':
            self.model.eval()
            with torch.no_grad():
                X = torch.FloatTensor(X_with_sentiment)
                predictions = self.model(X).numpy()
            return predictions.squeeze()
        else:
            X_2d = X_with_sentiment.reshape(X_with_sentiment.shape[0], -1)
            return self.model.predict(X_2d) 