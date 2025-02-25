# Quantrade: AI-Driven Stock Price Prediction

An advanced stock price prediction system that combines LSTM deep learning with sentiment analysis of financial news to improve forecasting accuracy.

## Features

- AI-driven stock price prediction using LSTM neural networks
- Real-time financial news sentiment analysis using NLTK VADER
- 25% improved forecasting accuracy through news integration
- Interactive web interface with real-time predictions
- Historical data caching with MongoDB
- Secure data storage with AWS S3

## Tech Stack

- Python
- PyTorch (LSTM implementation)
- scikit-learn (Data preprocessing)
- yfinance (Stock data)
- Flask (Web interface)
- MongoDB (News caching)
- AWS S3 (Data storage)
- NLTK (Sentiment analysis)

## Setup

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env`:

```
NEWS_API_KEY=your_newsapi_key
MONGODB_URI=your_mongodb_uri
AWS_ACCESS_KEY=your_aws_access_key
AWS_SECRET_KEY=your_aws_secret_key
S3_BUCKET_NAME=your_s3_bucket_name
```

4. Run the application:

```bash
python app.py
```

The web interface will be available at `http://localhost:5000`.

## Usage

1. Enter a stock symbol (e.g., AAPL, GOOGL)
2. Specify the number of days for prediction
3. View predictions, sentiment analysis, and recent news
4. Monitor accuracy improvements through news integration

## Architecture

- `app.py`: Flask web application
- `utils/`:
  - `models.py`: LSTM and Random Forest implementations
  - `sentiment_analysis.py`: News fetching and sentiment analysis
  - `data_processing.py`: Data preparation and preprocessing

## License

MIT License
