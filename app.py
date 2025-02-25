from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta
import pandas as pd
from utils.data_processing import fetch_stock_data, prepare_data, inverse_transform_predictions
from utils.models import StockPredictor
from utils.sentiment_analysis import NewsAnalyzer

app = Flask(__name__)

# Initialize models
predictor = StockPredictor(model_type='lstm')
news_analyzer = NewsAnalyzer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symbol = data.get('symbol', 'AAPL')
    days = int(data.get('days', 30))
    
    try:
        # Fetch historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)
        df = fetch_stock_data(symbol, start_date.strftime('%Y-%m-%d'), 
                            end_date.strftime('%Y-%m-%d'))
        
        # Prepare data
        X_train, X_test, y_train, y_test, scaler = prepare_data(df)
        
        # Train model
        losses = predictor.train(X_train, y_train, symbol)
        
        # Make predictions
        predictions = predictor.predict(X_test, symbol)
        
        # Transform predictions back to original scale
        actual_prices = inverse_transform_predictions(y_test, scaler)
        predicted_prices = inverse_transform_predictions(predictions, scaler)
        
        # Get sentiment data
        articles = news_analyzer.fetch_financial_news(symbol)
        sentiment_score = news_analyzer.get_aggregated_sentiment(articles)
        
        # Calculate accuracy improvement
        baseline_mse = ((actual_prices - actual_prices.mean()) ** 2).mean()
        model_mse = ((actual_prices - predicted_prices) ** 2).mean()
        accuracy_improvement = ((baseline_mse - model_mse) / baseline_mse) * 100
        
        return jsonify({
            'success': True,
            'predictions': predicted_prices.tolist(),
            'actual_prices': actual_prices.tolist(),
            'sentiment_score': sentiment_score,
            'accuracy_improvement': accuracy_improvement
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/sentiment', methods=['GET'])
def get_sentiment():
    symbol = request.args.get('symbol', 'AAPL')
    articles = news_analyzer.fetch_financial_news(symbol)
    sentiment_score = news_analyzer.get_aggregated_sentiment(articles)
    
    return jsonify({
        'success': True,
        'sentiment_score': sentiment_score,
        'articles': articles
    })

if __name__ == '__main__':
    app.run(debug=True) 