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
            if not chunk_data.empty:
                all_data.append(chunk_data)
            
            # Add delay between chunks to avoid rate limiting
            time.sleep(1.0)
        
        if all_data:
            # Combine all chunks
            try:
                return pd.concat(all_data, axis=1)
            except Exception as e:
                if VERBOSE:
                    print(f"Error combining chunks: {e}")
                return pd.DataFrame()
        else:
            return pd.DataFrame()
    
    # For smaller numbers of symbols, try downloading individually if batch download fails
    for attempt in range(max_retries):
        try:
            # Download data with timeout
            df = yf.download(
                symbols,
                start=start_date,
                end=end_date,
                progress=False,
                actions=False,
                threads=False,  # Single-threaded to reduce rate limiting issues
                timeout=timeout  # Add timeout to prevent hanging
            )
            
            if df.empty:
                if attempt < max_retries - 1:
                    time.sleep(delay * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    # If batch download failed, try individual downloads
                    if len(symbols) > 1 and attempt == max_retries - 1:
                        if VERBOSE:
                            print("Batch download failed, trying individual downloads...")
                        return download_individual_symbols(symbols, start_date, end_date, max_retries//2, delay, timeout)
                    return pd.DataFrame()
            
            return df
            
        except Exception as e:
            if "Rate limit" in str(e) and attempt < max_retries - 1:
                if VERBOSE:
                    print(f"Rate limit hit, retrying in {delay * (attempt + 1)} seconds...")
                time.sleep(delay * (attempt + 1))  # Exponential backoff
            else:
                if VERBOSE:
                    print(f"Error downloading data: {e}")
                # If batch download failed with an error, try individual downloads
                if len(symbols) > 1 and attempt == max_retries - 1:
                    if VERBOSE:
                        print("Batch download failed with error, trying individual downloads...")
                    return download_individual_symbols(symbols, start_date, end_date, max_retries//2, delay, timeout)
                return pd.DataFrame()
    
    return pd.DataFrame()

def download_individual_symbols(symbols, start_date=None, end_date=None, max_retries=3, delay=1.0, timeout=15):
    """Helper function to download symbols individually with timeout"""
    result_dfs = []
    
    for symbol in symbols:
        if VERBOSE:
            print(f"Downloading individual symbol: {symbol}")
        
        try:
            # Download this single symbol with timeout
            single_df = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                progress=False,
                actions=False,
                threads=False,
                timeout=timeout  # Add timeout to prevent hanging
            )
            
            if not single_df.empty:
                # For a single symbol, yfinance doesn't create MultiIndex columns
                # So we need to create one to match the batch download format
                columns = pd.MultiIndex.from_product([single_df.columns, [symbol]])
                single_df.columns = columns
                result_dfs.append(single_df)
            
            # Add a short delay to avoid rate limiting
            time.sleep(0.5)
        
        except Exception as e:
            if VERBOSE:
                print(f"Error downloading individual symbol {symbol}: {e}")
            # Just skip this symbol and continue
            continue
    
    if result_dfs:
        try:
            # Combine all the individual results
            return pd.concat(result_dfs, axis=1)
        except Exception as e:
            if VERBOSE:
                print(f"Error combining individual results: {e}")
    
    return pd.DataFrame()

def load_from_cache(cache_key):
    """Load data from cache if available and valid"""
    if not os.path.exists(CACHE_DIR):
        return None
        
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
    
    if not os.path.exists(cache_file):
        return None
        
    # Check if cache is still valid
    file_time = os.path.getmtime(cache_file)
    file_age = (time.time() - file_time) / (60 * 60 * 24)  # Age in days
    
    if file_age > CACHE_VALID_DAYS:
        return None
        
    try:
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        if VERBOSE:
            print(f"Error loading cache: {e}")
        return None

def save_to_cache(cache_key, data):
    """Save data to cache"""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
    
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        if VERBOSE:
            print(f"Error saving to cache: {e}")

def calculate_momentum_scores(start_date=None, end_date=None, use_cache=True, custom_file=None):
    """
    Calculate momentum scores for stocks
    
    Parameters:
    -----------
    start_date : str, optional
        Start date for data in YYYY-MM-DD format
    end_date : str, optional
        End date for data in YYYY-MM-DD format
    use_cache : bool, optional
        Whether to use cached data if available
    custom_file : UploadedFile, optional
        A custom CSV file uploaded by the user with ticker symbols
        
    Returns:
    --------
    dict
        Dictionary containing DataFrame with momentum scores and other metadata
    """
    # Debug logging to verify data source
    using_custom = custom_file is not None
    if VERBOSE:
        print(f"calculate_momentum_scores called with custom_file: {using_custom}")
        print(f"Cache enabled: {use_cache}")
        if using_custom:
            try:
                custom_file.seek(0)
                print(f"Custom file position reset to beginning")
            except Exception as e:
                print(f"Error resetting file position: {e}")
    # Set date range
    if not end_date:
        end_date = date.today().isoformat()
    if not start_date:
        # Use 2 years of data to ensure enough history
        start_date = (date.today() - timedelta(days=730)).isoformat()
    
    # Generate cache key based on date range and file source
    file_identifier = "custom" if custom_file is not None else "sp500"
    
    # If using custom file, add a timestamp to the cache key to ensure uniqueness
    if custom_file is not None:
        custom_timestamp = str(int(time.time()))
        file_identifier = f"{file_identifier}_{custom_timestamp}"
        
    cache_key = f"momentum_{file_identifier}_{start_date}_{end_date}"
    
    # Try to load from cache first if enabled
    if use_cache:
        cached_result = load_from_cache(cache_key)
        if cached_result is not None:
            if VERBOSE:
                print("Using cached momentum data")
            return cached_result
    
    # Load tickers from the provided file or default S&P 500 file
    from data_loader import load_ticker_data
    df_tickers = load_ticker_data(custom_file)
    symbols_all = df_tickers["Symbol"].unique().tolist()
    
    # Fix any ticker naming issues and filter out unwanted symbols
    replacements = {"BRK.B": "BRK-B"}
    symbols_fixed = [replacements.get(sym, sym) for sym in symbols_all]
    symbols_fixed = [s for s in symbols_fixed if s not in ("ATVI",)]
    symbols_all = sorted(set(symbols_fixed))

    if VERBOSE:
        print("Processing symbols:")
        print(f"Total symbols: {len(symbols_all)}")
    
    # Use smaller chunks and add delay between downloads to avoid rate limiting
    chunk_size = 25  # Smaller chunks
    num_chunks = math.ceil(len(symbols_all) / chunk_size)
    
    all_chunks = []
    successful_symbols = []
    
    for i in range(num_chunks):
        chunk_symbols = symbols_all[i*chunk_size:(i+1)*chunk_size]
        if VERBOSE:
            print(f"\nDownloading chunk {i+1}/{num_chunks} ({len(chunk_symbols)} symbols)")
        
        # Use our retry function with timeout
        df_chunk = download_data_with_retry(
            chunk_symbols,
            start_date=start_date,
            end_date=end_date,
            timeout=30  # Set a 30 second timeout for API requests
        )
        
        if df_chunk.empty:
            if VERBOSE:
                print(f"No data for chunk {i+1}, skipping...")
            continue
        
        # Extract close prices and convert to long format
        try:
            df_close = df_chunk["Close"]
            # Handle both single-symbol and multi-symbol cases
            if len(chunk_symbols) == 1:
                df_long = pd.DataFrame({
                    'date': df_close.index,
                    'symbol': chunk_symbols[0],
                    'close': df_close.values
                })
                successful_symbols.append(chunk_symbols[0])
            else:
                # Get the actual symbols that were successfully downloaded
                available_symbols = df_close.columns.tolist()
                successful_symbols.extend(available_symbols)
                
                df_long = df_close.reset_index()
                df_long = pd.melt(df_long, id_vars=['Date'], value_vars=available_symbols, 
                                 var_name='symbol', value_name='close')
                df_long.rename(columns={'Date': 'date'}, inplace=True)
            
            df_long["date"] = pd.to_datetime(df_long["date"])
            all_chunks.append(df_long)
        except Exception as e:
            if VERBOSE:
                print(f"Error processing chunk {i+1}: {e}")
        
        # Add delay between chunks to avoid rate limiting
        if i < num_chunks - 1:
            time.sleep(1.0)
    
    # If we have no data at all, return an error
    if not all_chunks:
        return {
            "error": "Failed to download any stock data. This could be due to API rate limiting or connectivity issues."
        }
    
    # Even if we only got some data, proceed with what we have
    data_long = pd.concat(all_chunks, ignore_index=True)
    
    if VERBOSE:
        print(f"\nSuccessfully downloaded data for {len(successful_symbols)} out of {len(symbols_all)} symbols")
    
    data_long["date"] = pd.to_datetime(data_long["date"])
    data_long = data_long.dropna(subset=["date", "close"])
    data_long = data_long.sort_values(["symbol", "date"])
    
    if VERBOSE:
        print("\nData range:")
        print(f"Start date: {data_long['date'].min()}")
        print(f"End date: {data_long['date'].max()}")
        print(f"Number of symbols with data: {data_long['symbol'].nunique()}")
    
    # Filter for symbols with sufficient history
    min_observations = 126  # Reduced to 6 months to ensure we have enough stocks
    symbol_counts = data_long.groupby("symbol").size()
    valid_symbols = symbol_counts[symbol_counts >= min_observations].index
    data_filtered = data_long[data_long.symbol.isin(valid_symbols)].copy()
    
    excluded_symbols = sorted(set(symbols_all) - set(valid_symbols))
    if VERBOSE and excluded_symbols:
        print(f"\nExcluded {len(excluded_symbols)} symbols due to insufficient history")
    
    # If we have no valid symbols, return an error
    if len(valid_symbols) == 0:
        return {
            "error": "No symbols with sufficient price history. Try expanding the date range."
        }
    
    # Calculate momentum
    data_filtered = data_filtered.set_index(["symbol", "date"])
    prices = data_filtered[["close"]].sort_index()
    
    # Calculate momentum for each symbol with time limit
    momentum_values = []
    start_time = time.time()
    max_calculation_time = 1200  # 5 minutes maximum for the entire calculation
    
    symbols_to_process = list(prices.index.get_level_values("symbol").unique())
    total_symbols = len(symbols_to_process)
    processed_symbols = 0
    
    for symbol in symbols_to_process:
        # Check if we've exceeded the maximum calculation time
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        if elapsed_time > max_calculation_time:
            if VERBOSE:
                print(f"\nReached maximum calculation time of {max_calculation_time} seconds.")
                print(f"Processed {processed_symbols}/{total_symbols} symbols ({processed_symbols/total_symbols*100:.1f}%)")
            break
        
        if VERBOSE:
            print(f"\nProcessing momentum for {symbol} ({processed_symbols+1}/{total_symbols})")
        
        try:
            symbol_data = prices.loc[symbol]["close"]
            
            # Create an empty Series with a specified dtype to avoid warnings
            momentum_series = pd.Series(index=symbol_data.index, dtype="float64")
            
            # If we have enough data for regular calculation
            if len(symbol_data) >= 252:
                for i in range(252, len(symbol_data)):
                    window = symbol_data.iloc[i-252:i+1]
                    momentum_series.iloc[i] = momentum(window)
            # Use a smaller window if we don't have enough data but at least 126 data points
            elif len(symbol_data) >= 126:
                window_size = min(126, len(symbol_data) - 1)
                for i in range(window_size, len(symbol_data)):
                    window = symbol_data.iloc[i-window_size:i+1]
                    # Use adapted momentum calculation for smaller window
                    momentum_value = (window.iloc[-1] - window.iloc[0]) / window.iloc[0]
                    momentum_series.iloc[i] = momentum_value
            
            # Convert to DataFrame and assign a MultiIndex
            momentum_df = pd.DataFrame(momentum_series, columns=["momentum"])
            momentum_df.index = pd.MultiIndex.from_product(
                [[symbol], momentum_df.index],
                names=["symbol", "date"]
            )
            momentum_values.append(momentum_df)
            processed_symbols += 1
        except Exception as e:
            if VERBOSE:
                print(f"Error calculating momentum for {symbol}: {e}")
            processed_symbols += 1
    
    # Check if we have any momentum values
    if not momentum_values:
        return {
            "error": "Failed to calculate momentum for any stocks. Check data quality and availability."
        }
    
    df_momentum = pd.concat(momentum_values)
    
    if VERBOSE:
        print("\nMomentum calculation summary:")
        print("Total momentum values:", len(df_momentum))
        print("Non-NaN momentum values:", df_momentum["momentum"].notna().sum())
    
    # Combine price and momentum data
    combined = prices.join(df_momentum, how="left")
    
    # Get dates with valid momentum values
    valid_dates = combined[combined["momentum"].notna()].index.get_level_values("date").unique()
    
    if len(valid_dates) > 0:
        # Rank stocks by momentum for each date
        combined["factor_rank"] = combined.groupby(level="date")["momentum"]\
                                       .rank(ascending=False, method="first")
        
        # Get the most recent date with valid momentum values
        last_date = valid_dates[-1]
        
        today_data = combined.xs(last_date, level="date").dropna()
        today_sorted = today_data.sort_values("factor_rank")
        
        # Prepare result
        result = {
            "momentum_data": df_momentum,
            "combined_data": combined,
            "last_date": last_date,
            "today_sorted": today_sorted,
            "valid_dates": valid_dates,
            "tickers_info": df_tickers,
            "excluded_symbols": excluded_symbols
        }
        
        # Cache the result for future use
        if use_cache:
            save_to_cache(cache_key, result)
        
        return result
    else:
        return {
            "error": "No valid momentum values found. Check the momentum calculation and data quality."
        }

def get_top_bottom_stocks(momentum_result, n=5):
    """Get the top and bottom N stocks by momentum"""
    if "error" in momentum_result:
        return None
    
    today_sorted = momentum_result["today_sorted"]
    # Select top and bottom N stocks
    top_N = min(n, len(today_sorted) // 2)
    stocks_to_buy = today_sorted.head(top_N).assign(side=1)
    stocks_to_short = today_sorted.tail(top_N).assign(side=-1)
    stocks_to_trade = pd.concat([stocks_to_buy, stocks_to_short])
    
    return stocks_to_trade
