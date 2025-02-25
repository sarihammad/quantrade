import os
from typing import List, Dict
from datetime import datetime, timedelta
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from newsapi import NewsAPI
import boto3
from pymongo import MongoClient
from dotenv import load_dotenv

# Download required NLTK data
nltk.download('vader_lexicon')

class NewsAnalyzer:
    def __init__(self):
        load_dotenv()
        
        # Initialize NewsAPI
        self.newsapi = NewsAPI(os.getenv('NEWS_API_KEY'))
        
        # Initialize MongoDB
        mongo_uri = os.getenv('MONGODB_URI')
        self.mongo_client = MongoClient(mongo_uri)
        self.db = self.mongo_client['quantrade']
        self.news_collection = self.db['financial_news']
        
        # Initialize AWS S3
        self.s3 = boto3.client('s3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY'),
            aws_secret_access_key=os.getenv('AWS_SECRET_KEY')
        )
        self.bucket_name = os.getenv('S3_BUCKET_NAME')
        
        # Initialize sentiment analyzer
        self.sia = SentimentIntensityAnalyzer()
    
    def fetch_financial_news(self, symbol: str, days: int = 7) -> List[Dict]:
        """Fetch financial news for a given stock symbol"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Check if we have cached news in MongoDB
        cached_news = list(self.news_collection.find({
            'symbol': symbol,
            'date': {'$gte': start_date, '$lte': end_date}
        }))
        
        if cached_news:
            return cached_news
        
        # Fetch new articles from NewsAPI
        query = f"{symbol} stock market"
        articles = self.newsapi.get_everything(
            q=query,
            from_param=start_date.strftime('%Y-%m-%d'),
            to=end_date.strftime('%Y-%m-%d'),
            language='en',
            sort_by='publishedAt'
        )
        
        # Process and store articles
        processed_articles = []
        for article in articles['articles']:
            processed_article = {
                'symbol': symbol,
                'date': datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ'),
                'title': article['title'],
                'description': article['description'],
                'sentiment_scores': self.analyze_sentiment(f"{article['title']} {article['description']}")
            }
            processed_articles.append(processed_article)
            
            # Store in MongoDB
            self.news_collection.insert_one(processed_article)
        
        return processed_articles
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of text using VADER"""
        return self.sia.polarity_scores(text)
    
    def get_aggregated_sentiment(self, articles: List[Dict]) -> float:
        """Calculate aggregated sentiment score from articles"""
        if not articles:
            return 0.0
        
        compound_scores = [article['sentiment_scores']['compound'] for article in articles]
        return sum(compound_scores) / len(compound_scores)
    
    def save_sentiment_data(self, symbol: str, sentiment_data: pd.DataFrame):
        """Save sentiment data to S3"""
        csv_buffer = sentiment_data.to_csv(index=False)
        s3_key = f'sentiment_data/{symbol}/{datetime.now().strftime("%Y-%m-%d")}.csv'
        
        self.s3.put_object(
            Bucket=self.bucket_name,
            Key=s3_key,
            Body=csv_buffer
        ) 