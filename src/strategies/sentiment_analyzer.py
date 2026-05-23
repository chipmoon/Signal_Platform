"""
News Sentiment Analyzer
=======================
Fetches and analyzes market-specific news to generate a sentiment bias.
Acts as a regime filter for the trading system.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

try:
    import yfinance as yf
except ImportError:
    yf = None


class SentimentAnalyzer:
    """
    Analyzes news sentiment for a given ticker or market.
    Currently uses Yahoo Finance News as a baseline source.
    """

    def __init__(self, threshold: float = 0.2):
        """
        Initialize the Sentiment Analyzer.
        
        Args:
            threshold: Minimum sentiment score to allow a trade (default: 0.2)
        """
        self.threshold = threshold
        # Positive/Negative keyword map (Simple NLP fallback)
        self.keywords = {
            "bullish": 0.5, "upbeat": 0.4, "growth": 0.3, "higher": 0.3, "record": 0.4,
            "bearish": -0.5, "fear": -0.4, "recession": -0.6, "lower": -0.3, "drop": -0.4,
            "war": -0.7, "hike": -0.3, "crisis": -0.6, "inflation": -0.2
        }

    def get_sentiment(self, symbol: str) -> float:
        """
        Fetch news for a symbol and calculate aggregate sentiment.
        
        Returns:
            Float between -1.0 (extremely bearish) and 1.0 (extremely bullish)
        """
        logger.info(f"Analyzing sentiment for {symbol}...")
        if yf is None:
            logger.warning("yfinance is not installed. Sentiment defaults to neutral.")
            return 0.0
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            if not news:
                logger.warning(f"No news found for {symbol}. Returning neutral sentiment.")
                return 0.0
            
            total_sentiment = 0.0
            count = 0
            
            for article in news[:10]: # Analyze latest 10 articles
                title = article.get("title", "").lower()
                score = 0.0
                
                # Check keywords in title
                for word, weight in self.keywords.items():
                    if word in title:
                        score += weight
                
                # Normalize and clip
                score = max(min(score, 1.0), -1.0)
                total_sentiment += score
                count += 1
            
            avg_sentiment = total_sentiment / count if count > 0 else 0.0
            
            # Add a small random noise to simulate 'regime shifts' in backtests
            # if we are using historical data where real news dates aren't easily syncable
            logger.success(f"Sentiment for {symbol}: {avg_sentiment:.2f}")
            return avg_sentiment
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed for {symbol}: {e}")
            return 0.0

    def generate_sentiment_series(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Injects a sentiment column into a price DataFrame.
        Note: For historical simulation without a proper news database, 
        this uses the current headline bias as a persistent state.
        """
        df = data.copy()
        current_bias = self.get_sentiment(symbol)
        
        # In a real system, we'd have news per Date. 
        # For this high-level integration, we propagate the current headline bias.
        df["sentiment_score"] = current_bias
        
        # Determine sentiment bias: 1 (Bullish), -1 (Bearish), 0 (Neutral)
        df["sentiment_bias"] = 0
        df.loc[df["sentiment_score"] >= self.threshold, "sentiment_bias"] = 1
        df.loc[df["sentiment_score"] <= -self.threshold, "sentiment_bias"] = -1
        
        return df
