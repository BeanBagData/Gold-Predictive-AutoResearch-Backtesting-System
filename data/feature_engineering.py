"""
data/feature_engineering.py
===========================
Vectorized calculation of the primary short-term TA stack.
Target shifting dynamically adapts based on the user's --horizon setting.
"""
import pandas as pd
import numpy as np

def build_features(df: pd.DataFrame, p, target_bars: int = 1) -> pd.DataFrame:
    """Computes all requested indicators efficiently using Pandas/NumPy."""
    df = df.copy()

    # 1. EMA
    df['ema_f'] = df['close'].ewm(span=p.ema_fast, adjust=False).mean()
    df['ema_s'] = df['close'].ewm(span=p.ema_slow, adjust=False).mean()

    # 2. RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/p.rsi_period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/p.rsi_period, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))

    # 3. MACD
    macd_f = df['close'].ewm(span=p.macd_fast, adjust=False).mean()
    macd_s = df['close'].ewm(span=p.macd_slow, adjust=False).mean()
    df['macd'] = macd_f - macd_s
    df['macd_sig'] = df['macd'].ewm(span=p.macd_signal, adjust=False).mean()

    # 4. Stochastic
    low_min = df['low'].rolling(p.stoch_k).min()
    high_max = df['high'].rolling(p.stoch_k).max()
    df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-9)
    df['stoch_d'] = df['stoch_k'].rolling(p.stoch_d).mean()

    # 5. ATR
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift()).abs()
    tr3 = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = tr.ewm(span=p.atr_period, adjust=False).mean()
    df['atr_sma'] = df['atr'].rolling(window=p.atr_period*2).mean() # Baseline volatility

    # 6. ADX
    up_move = df['high'] - df['high'].shift()
    down_move = df['low'].shift() - df['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/p.adx_period, adjust=False).mean() / (df['atr'] + 1e-9)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/p.adx_period, adjust=False).mean() / (df['atr'] + 1e-9)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    df['adx'] = dx.ewm(alpha=1/p.adx_period, adjust=False).mean()

    # 7. Bollinger Bands
    sma = df['close'].rolling(p.bb_period).mean()
    std = df['close'].rolling(p.bb_period).std()
    df['bb_up'] = sma + p.bb_stdev * std
    df['bb_dn'] = sma - p.bb_stdev * std

    # TARGET: Will close[t + target_bars] be above or below close[t]?
    # Must use single future close — EA places orders relative to current close,
    # so the prediction reference point must match exactly.
    df['future_close'] = df['close'].shift(-target_bars)
    df['actual_up'] = df['future_close'] > df['close']
    df['actual_down'] = df['future_close'] < df['close']

    return df.dropna()