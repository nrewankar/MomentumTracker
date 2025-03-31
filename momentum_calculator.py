import math
import numpy as np
import pandas as pd
import yfinance as yf
import time
import os
import pickle
from datetime import date, timedelta

# Set a flag to control verbosity
VERBOSE = True

# Cache configuration
CACHE_DIR = ".cache"
CACHE_VALID_DAYS = 1  # Consider cache valid for 1 day

def momentum(close_series):
    """
    Computes momentum over a rolling 252-day window:
      - Long-term return (252 days)
      - Short-term return (21 days)
      - Normalized by stdev of daily returns over 126 days
    """
    # Check if we have enough data points
    if len(close_series) < 252:
        return np.nan
        
    # Daily returns over last 126 days
    returns_126 = pd.Series(close_series).pct_change().iloc[-126:]
    stdev_126 = returns_126.std()
    
    if pd.isna(stdev_126) or stdev_126 == 0:
        return np.nan
        
    try:
        # Long-term return (252 days)
        long_term = (close_series.iloc[-1] - close_series.iloc[0]) / close_series.iloc[0]
        # Short-term return (21 days)
        short_term = (close_series.iloc[-1] - close_series.iloc[-21]) / close_series.iloc[-21]
        result = (long_term - short_term) / stdev_126
        return result
    except Exception as e:
        return np.nan

def download_data_with_retry(symbols, start_date=None, end_date=None, max_retries=5, delay=2.0, timeout=30):
    """
    Download data with retry logic to handle rate limiting
    
    Parameters:
    -----------
    symbols : list
        List of ticker symbols to download
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    max_retries : int
        Maximum number of retry attempts
    delay : float
        Delay between retries in seconds
    timeout : int
        Timeout in seconds for API request
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with downloaded stock data
    """
    # For large batches, break into smaller chunks for better reliability
    if len(symbols) > 50:
        # Break into chunks of 25
        chunk_size = 25
        all_data = []
        for i in range(0, len(symbols), chunk_size):
            chunk_symbols = symbols[i:i+chunk_size]
            if VERBOSE:
                print(f"Processing chunk {i//chunk_size + 1}/{math.ceil(len(symbols)/chunk_size)} with {len(chunk_symbols)} symbols")
            
            chunk_data = download_data_with_retry(chunk_symbols, start_date, end_date, max_retries, delay)
