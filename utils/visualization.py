import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Optional

def plot_stock_prediction(actual_values: np.ndarray, 
                         predicted_values: np.ndarray,
                         dates: Optional[List] = None,
                         title: str = "Stock Price Prediction"):
    """
    Plot actual vs predicted stock prices
    """
    plt.figure(figsize=(15, 6))
    if dates is None:
        plt.plot(actual_values, label='Actual', color='blue')
        plt.plot(predicted_values, label='Predicted', color='red', linestyle='--')
    else:
        plt.plot(dates, actual_values, label='Actual', color='blue')
        plt.plot(dates, predicted_values, label='Predicted', color='red', linestyle='--')
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
def plot_training_loss(losses: List[float]):
    """
    Plot training loss over epochs
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    
def plot_prediction_error(actual_values: np.ndarray, 
                         predicted_values: np.ndarray):
    """
    Plot prediction error distribution
    """
    errors = actual_values - predicted_values
    
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.title('Prediction Error Distribution')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout() 