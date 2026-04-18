"""
backtest/gold_backtester.py
===========================
Vectorized TA predictive scorer.
Logic optimized to maintain sufficient signal volume across various prediction horizons.
"""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np

from data.feature_engineering import build_features

@dataclass
class PredictorParams:
    # Trend
    ema_fast: int = 10
    ema_slow: int = 20

    # Momentum
    rsi_period: int = 10
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0

    # MACD
    macd_fast: int = 5
    macd_slow: int = 13
    macd_signal: int = 1

    # Trend strength
    adx_period: int = 10
    adx_thresh: float = 20.0

    # Timing
    stoch_k: int = 14
    stoch_d: int = 3
    stoch_smooth: int = 3

    # Volatility
    bb_period: int = 20
    bb_stdev: float = 2.0
    atr_period: int = 14
    atr_mult: float = 1.0


@dataclass
class PredictionResult:
    total_signals: int
    correct_signals: int
    incorrect_signals: int
    win_rate: float
    up_signals: int
    down_signals: int
    up_win_rate: float
    down_win_rate: float
    max_consec_correct: int
    max_consec_incorrect: int
    win_rate_stability: float
    signal_series: pd.Series
    result_series: pd.Series


def run_predictive_backtest(df: pd.DataFrame, params: PredictorParams, target_bars: int = 1) -> PredictionResult:
    """Vectorized calculation of signal alignment with the subsequent target bar."""
    
    # 1. Compute dynamic indicators and targets
    data = build_features(df, params, target_bars)

    # 2. Consensus Voting Logic (Optimized for Statistical Significance)
    trend_up = (data['ema_f'] > data['ema_s'])
    trend_dn = (data['ema_f'] < data['ema_s'])

    mom_up = (data['macd'] > data['macd_sig']) | (data['stoch_k'] > data['stoch_d'])
    mom_dn = (data['macd'] < data['macd_sig']) | (data['stoch_k'] < data['stoch_d'])

    vol_active = (data['adx'] > params.adx_thresh) | (data['atr'] > (data['atr_sma'] * params.atr_mult))

    safe_up = (data['close'] < data['bb_up']) & (data['rsi'] < params.rsi_overbought)
    safe_dn = (data['close'] > data['bb_dn']) & (data['rsi'] > params.rsi_oversold)

    bullish_cond = trend_up & mom_up & vol_active & safe_up
    bearish_cond = trend_dn & mom_dn & vol_active & safe_dn

    # Convert to numeric signals: +1 (Up), -1 (Down), 0 (None)
    signals = pd.Series(0, index=data.index)
    signals.loc[bullish_cond] = 1
    signals.loc[bearish_cond] = -1

    # 3. Evaluate Predictions dynamically against target_bars
    evaluations = pd.Series(np.nan, index=data.index)
    
    correct_up = (signals == 1) & data['actual_up']
    evaluations.loc[signals == 1] = 0
    evaluations.loc[correct_up] = 1

    correct_down = (signals == -1) & data['actual_down']
    evaluations.loc[signals == -1] = 0
    evaluations.loc[correct_down] = 1

    # 4. Aggregate Metrics
    mask_active = signals != 0
    total_signals = mask_active.sum()
    
    if total_signals == 0:
        return PredictionResult(0,0,0,0.0,0,0,0.0,0.0,0,0, 0.0, signals, evaluations)

    correct_signals = (evaluations == 1).sum()
    incorrect_signals = (evaluations == 0).sum()
    win_rate = correct_signals / total_signals

    up_sigs = (signals == 1).sum()
    dn_sigs = (signals == -1).sum()
    up_wr = correct_up.sum() / up_sigs if up_sigs > 0 else 0.0
    dn_wr = correct_down.sum() / dn_sigs if dn_sigs > 0 else 0.0

    # Consecutive logic
    streak_correct = 0
    streak_incorrect = 0
    max_c_correct = 0
    max_c_incorrect = 0
    
    for val in evaluations.dropna():
        if val == 1:
            streak_correct += 1
            streak_incorrect = 0
            if streak_correct > max_c_correct: 
                max_c_correct = streak_correct
        else:
            streak_incorrect += 1
            streak_correct = 0
            if streak_incorrect > max_c_incorrect: 
                max_c_incorrect = streak_incorrect

    # Win-Rate Stability
    win_rate_stability = 0.0
    if total_signals > 50:
        rolling_wr_std = evaluations.dropna().rolling(window=25).mean().std()
        win_rate_stability = max(0.0, 1.0 - (rolling_wr_std * 3.0)) 

    return PredictionResult(
        total_signals=int(total_signals),
        correct_signals=int(correct_signals),
        incorrect_signals=int(incorrect_signals),
        win_rate=float(win_rate),
        up_signals=int(up_sigs),
        down_signals=int(dn_sigs),
        up_win_rate=float(up_wr),
        down_win_rate=float(dn_wr),
        max_consec_correct=int(max_c_correct),
        max_consec_incorrect=int(max_c_incorrect),
        win_rate_stability=float(win_rate_stability),
        signal_series=signals,
        result_series=evaluations
    )
