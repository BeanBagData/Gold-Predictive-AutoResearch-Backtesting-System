"""
data/loader.py 
==============
Standard MT5 CSV loader for XAUUSD / Forex data.
Handles tab-separated or comma-separated exports.
"""
import pandas as pd
import os

def load_mt5_csv(path: str) -> pd.DataFrame:
    """
    Loads MT5 OHLCV CSV. 
    Expects columns: <DATE>, <TIME>, <OPEN>, <HIGH>, <LOW>, <CLOSE>, <TICKVOL>
    """
    # Detect separator (MT5 usually uses tabs \t)
    with open(path, 'r') as f:
        first_line = f.readline()
        sep = '\t' if '\t' in first_line else ','

    df = pd.read_csv(path, sep=sep)
    
    # Clean up column names (remove < > and lower case)
    df.columns = [c.strip('<>').lower() for c in df.columns]
    
    # Standardize 'tickvol' to 'volume' if necessary
    if 'tickvol' in df.columns:
        df = df.rename(columns={'tickvol': 'volume'})
    
    # Combine Date and Time into a single index
    if 'date' in df.columns and 'time' in df.columns:
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        df = df.set_index('datetime')
    
    # Ensure standard column set exists
    required = ['open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            # Fallback for some MT5 formats that use 'vol'
            if col == 'volume' and 'vol' in df.columns:
                df = df.rename(columns={'vol': 'volume'})
            else:
                raise ValueError(f"CSV is missing required column: {col}")

    # Sort and convert to float
    df = df.sort_index()
    return df[required].astype(float)

def describe_data(df: pd.DataFrame) -> dict:
    """Helper to verify data integrity."""
    return {
        'total_bars': len(df),
        'start': df.index[0],
        'end': df.index[-1],
        'avg_spread_estimate': (df['high'] - df['low']).mean()
    }