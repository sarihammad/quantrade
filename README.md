# Stock Price Prediction using Machine Learning

This project implements a machine learning model to predict stock prices using historical data. It uses PyTorch for deep learning (LSTM) and traditional ML models like Random Forest for price prediction.

## Features

- Data fetching using Yahoo Finance API
- Data preprocessing and feature engineering
- LSTM model implementation using PyTorch
- Random Forest model implementation
- Model evaluation and comparison
- Visualization of predictions

## Setup

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

- `main.py`: Main script to run the stock prediction models
- `utils/`: Utility functions and helper modules
  - `data_processing.py`: Data loading and preprocessing functions
  - `models.py`: Model implementations (LSTM and Random Forest)
  - `visualization.py`: Functions for plotting results

## Usage

```bash
python main.py --symbol AAPL --model lstm --days 30
```

Arguments:

- `--symbol`: Stock symbol (default: AAPL)
- `--model`: Model type (lstm or rf) (default: lstm)
- `--days`: Number of days to predict (default: 30)

## Models

1. LSTM (Long Short-Term Memory)

   - Deep learning model for sequence prediction
   - Captures long-term dependencies in time series data

2. Random Forest
   - Ensemble learning method
   - Good for capturing non-linear relationships

## License

MIT License
