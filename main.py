import argparse
from datetime import datetime, timedelta
import torch
from utils.data_processing import fetch_stock_data, prepare_data, inverse_transform_predictions
from utils.models import StockPredictor
from utils.visualization import plot_stock_prediction, plot_training_loss, plot_prediction_error
import matplotlib.pyplot as plt

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Stock Price Prediction')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol')
    parser.add_argument('--model', type=str, default='lstm', choices=['lstm', 'rf'],
                        help='Model type (lstm or rf)')
    parser.add_argument('--days', type=int, default=30, help='Number of days to predict')
    args = parser.parse_args()
    
    # Set date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)  # 2 years of data
    
    # Fetch and prepare data
    print(f"Fetching data for {args.symbol}...")
    df = fetch_stock_data(args.symbol, start_date.strftime('%Y-%m-%d'), 
                         end_date.strftime('%Y-%m-%d'))
    
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    
    # Initialize and train model
    print(f"Training {args.model.upper()} model...")
    predictor = StockPredictor(model_type=args.model)
    losses = predictor.train(X_train, y_train)
    
    # Make predictions
    print("Making predictions...")
    predictions = predictor.predict(X_test)
    
    # Transform predictions back to original scale
    actual_prices = inverse_transform_predictions(y_test, scaler)
    predicted_prices = inverse_transform_predictions(predictions, scaler)
    
    # Plot results
    plot_stock_prediction(actual_prices, predicted_prices, 
                         title=f"{args.symbol} Stock Price Prediction")
    if args.model == 'lstm' and losses:
        plot_training_loss(losses)
    plot_prediction_error(actual_prices, predicted_prices)
    
    # Calculate and print metrics
    mse = ((actual_prices - predicted_prices) ** 2).mean()
    print(f"\nMean Squared Error: {mse:.2f}")
    
    plt.show()

if __name__ == "__main__":
    main() 