import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def load_ticker_data(custom_file=None):
    """
    Load ticker information from CSV file
    
    Parameters:
    -----------
    custom_file : UploadedFile, optional
        A custom CSV file uploaded by the user
        
    Returns:
    --------
    DataFrame
        DataFrame containing ticker information
    """
    try:
        if custom_file is not None:
            try:
                # Make sure we're at the beginning of the file
                custom_file.seek(0)
                print(f"Loading data from custom file")
            except Exception as e:
                print(f"Error resetting file position: {e}")
                
            # Read from uploaded file
            df = pd.read_csv(custom_file)
            print(f"Loaded {len(df)} rows from custom file")
        else:
            # Read from default S&P 500 file
            file_path = "stocks_sp500_current.csv"
            print(f"Loading data from S&P 500 file: {file_path}")
            df = pd.read_csv(file_path)
            print(f"Loaded {len(df)} rows from S&P 500 file")
            
        # Ensure the DataFrame has required columns
        required_columns = ["Symbol"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Uploaded file is missing required columns: {', '.join(missing_columns)}")
            
        # Add missing optional columns with default values if they don't exist
        if "Company" not in df.columns:
            df["Company"] = df["Symbol"]
        if "Industry" not in df.columns:
            df["Industry"] = "Unknown"
        if "Year_Added" not in df.columns:
            df["Year_Added"] = None
            
        return df
    except Exception as e:
        print(f"Error loading ticker data: {e}")
        return pd.DataFrame(columns=["Symbol", "Company", "Industry", "Year_Added"])

def get_industry_breakdown(df_tickers):
    """
    Calculate industry breakdown of S&P 500 stocks
    
    Parameters:
    -----------
    df_tickers : DataFrame
        DataFrame containing ticker information
        
    Returns:
    --------
    DataFrame
        DataFrame with industry counts and percentages
    """
    if df_tickers.empty:
        return pd.DataFrame()
        
    # Count stocks by industry
    industry_counts = df_tickers["Industry"].value_counts().reset_index()
    industry_counts.columns = ["Industry", "Count"]
    
    # Calculate percentages
    total = industry_counts["Count"].sum()
    industry_counts["Percentage"] = (industry_counts["Count"] / total * 100).round(1)
    
    return industry_counts.sort_values("Count", ascending=False)

def merge_company_info(momentum_df, tickers_df):
    """
    Merge momentum data with company information
    
    Parameters:
    -----------
    momentum_df : DataFrame
        DataFrame containing momentum scores
    tickers_df : DataFrame
        DataFrame containing company information
        
    Returns:
    --------
    DataFrame
        Merged DataFrame with momentum scores and company information
    """
    if momentum_df.empty or tickers_df.empty:
        return pd.DataFrame()
        
    # Reset index to get symbol as column
    if isinstance(momentum_df.index, pd.MultiIndex):
        reset_df = momentum_df.reset_index(level=0)
    else:
        reset_df = momentum_df.reset_index()
    
    # Merge with company info
    merged_df = pd.merge(
        reset_df,
        tickers_df,
        left_on="symbol",
        right_on="Symbol",
        how="left"
    )
    
    return merged_df

def format_momentum_data(momentum_results):
    """
    Format momentum data for display
    
    Parameters:
    -----------
    momentum_results : dict
        Dictionary containing momentum calculation results
        
    Returns:
    --------
    dict
        Dictionary containing formatted DataFrames for display
    """
    if not momentum_results or "error" in momentum_results:
        return {"error": momentum_results.get("error", "Failed to calculate momentum")}
    
    # Extract the sorted data for the most recent date
    today_sorted = momentum_results["today_sorted"].copy()
    
    # Reset index to get symbol as column
    today_df = today_sorted.reset_index()
    
    # Merge with company information
    tickers_df = momentum_results["tickers_info"]
    full_df = pd.merge(
        today_df,
        tickers_df,
        left_on="symbol",
        right_on="Symbol",
        how="left"
    )
    
    # Format columns for display
    display_df = full_df.copy()
    display_df["momentum"] = display_df["momentum"].round(4)
    display_df["factor_rank"] = display_df["factor_rank"].astype(int)
    
    # Create classifications
    num_stocks = len(display_df)
    display_df["classification"] = "Neutral"
    
    # Top 10%
    top_threshold = int(num_stocks * 0.1)
    display_df.loc[display_df["factor_rank"] <= top_threshold, "classification"] = "Strong Buy"
    display_df.loc[(display_df["factor_rank"] > top_threshold) & 
                  (display_df["factor_rank"] <= top_threshold*2), "classification"] = "Buy"
    
    # Bottom 10%
    bottom_threshold = int(num_stocks * 0.9)
    display_df.loc[display_df["factor_rank"] >= bottom_threshold, "classification"] = "Strong Sell"
    display_df.loc[(display_df["factor_rank"] < bottom_threshold) & 
                  (display_df["factor_rank"] >= bottom_threshold-top_threshold), "classification"] = "Sell"
    
    # Prepare result
    result = {
        "display_df": display_df,
        "last_date": momentum_results["last_date"],
        "industry_breakdown": get_industry_breakdown(tickers_df),
        "top_stocks": display_df[display_df["classification"] == "Strong Buy"].head(10),
        "bottom_stocks": display_df[display_df["classification"] == "Strong Sell"].head(10)
    }
    
    return result
