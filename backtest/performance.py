"""
backtest/performance.py
=======================
Classification and Predictive Accuracy reporting with Correlation Analysis.
"""
from __future__ import annotations
import json
import pandas as pd
from backtest.gold_backtester import PredictionResult

### MODIFIED ###: Added an optional title to make reports distinct (IS vs OOS)
def generate_report(result: PredictionResult, df: pd.DataFrame, target_bars: int = 1, title: str = "Strategy Report") -> str:
    """Basic backtest report for single runs."""
    w = 75
    if result.total_signals == 0:
        return f"{'='*w}\n {title.upper()}: NO SIGNALS GENERATED \n{'='*w}"

    days = len(df) / 288 
    signals_per_day = result.total_signals / max(1, days)
    target_min = target_bars * 5

    lines =[
        "=" * w,
        f" {title.upper()}",
        "=" * w,
        f"  Period          : {df.index[0]}  →  {df.index[-1]}",
        f"  Total Bars      : {len(df):,}",
        f"  Trading Days    : {days:.1f}",
        "-" * w,
        f"  PREDICTIVE ACCURACY ({target_min}M Ahead)",
        f"    Total Signals       : {result.total_signals:,} ({signals_per_day:.1f} per day)",
        f"    Correct Predictions : {result.correct_signals:,}",
        f"    Incorrect Preds     : {result.incorrect_signals:,}",
        f"    OVERALL WIN RATE    : {result.win_rate * 100:.2f}%",
        f"    Win-Rate Stability  : {result.win_rate_stability * 100:.1f}%", ### NEW ###
        "",
        "  DIRECTIONAL BREAKDOWN",
        f"    Long (Up) Signals   : {result.up_signals:,}  (Win Rate: {result.up_win_rate * 100:.2f}%)",
        f"    Short (Down) Signals: {result.down_signals:,}  (Win Rate: {result.down_win_rate * 100:.2f}%)",
        "",
        "  CONSECUTIVE STREAKS",
        f"    Max Consec Correct  : {result.max_consec_correct}",
        f"    Max Consec Incorrect: {result.max_consec_incorrect}",
        "-" * w,
    ]
    return "\n".join(lines)

def generate_meta_analysis_text(history: list, param_space: dict, best_params: dict) -> str:
    """Extracts meta-analysis, correlations, and Top/Worst examples to feed the LLM."""
    if not history:
        return "No history available yet."

    df_hist = pd.DataFrame(history)
    df_valid = df_hist[df_hist['total_signals'] > 0]
    
    lines = [
        f"  Total Experiments Conducted : {len(history)}",
        f"  Valid Iterations (Signals>0): {len(df_valid)}",
        "",
        "  CURRENT BEST DISCOVERED PARAMETERS (from In-Sample training):",
        "  -------------------------------------------------------------------------",
    ]
    
    for k, v in best_params.items():
        lines.append(f"    {k:<18} : {v}")

    lines.extend([
        "",
        "  PARAMETER CORRELATION TO IN-SAMPLE WIN-RATE (Pearson)",
        "  -------------------------------------------------------------------------",
        "  Positive (+) = Higher parameter value correlates to higher Win Rate.",
        "  Negative (-) = Lower parameter value correlates to higher Win Rate.",
        "  -------------------------------------------------------------------------",
    ])
    
    # ### MODIFIED ###: Added win_rate_stability to correlation check
    if len(df_valid) > 5 and df_valid['win_rate'].std() > 0:
        corrs = df_valid.corr(numeric_only=True)['win_rate'].drop(
            ['score', 'win_rate', 'total_signals', 'generation', 'win_rate_stability'], 
            errors='ignore'
        )
        for k, c in corrs.sort_values(ascending=False).items():
            if pd.isna(c): continue
            
            strength = "Neutral"
            if abs(c) > 0.4: strength = "Strong"
            elif abs(c) > 0.2: strength = "Moderate"
            elif abs(c) > 0.1: strength = "Weak"
            
            direction = "Positive" if c > 0 else "Negative"
            label = f"({strength} {direction})" if strength != "Neutral" else "(Neutral)"
            lines.append(f"    {k:<18} : {c:>6.2f}  {label}")
    else:
        lines.append("    Not enough valid variance to calculate correlations.")

    lines.extend([
        "",
        "  PARAMETER SPACE EXPLORATION COVERAGE",
        "  -------------------------------------------------------------------------",
    ])

    for k, (lo, hi, _) in param_space.items():
        if k in df_hist.columns:
            t_min = df_hist[k].min()
            t_max = df_hist[k].max()
            coverage = (t_max - t_min) / max(1e-9, (hi - lo)) * 100
            
            warning = ""
            best_val = best_params.get(k, 0)
            if best_val <= lo + (hi-lo)*0.05:
                warning = "<< AT MIN BOUND"
            elif best_val >= hi - (hi-lo)*0.05:
                warning = ">> AT MAX BOUND"

            lines.append(f"    {k:<18} | Allowed: [{lo:g}, {hi:g}] | Tested: [{t_min:.1f}, {t_max:.1f}] | Cov: {coverage:3.0f}%  {warning}")

    if len(df_valid) >= 10:
        df_best = df_valid.nlargest(5, 'score')
        df_worst = df_valid.nsmallest(5, 'score')
        param_keys = list(param_space.keys())

        lines.extend(["", "  TOP 5 BEST IN-SAMPLE CONFIGURATIONS:", "  -------------------------------------------------------------------------"])
        for idx, row in df_best.iterrows():
            p_dict = {k: row[k] for k in param_keys if k in row}
            lines.append(f"    Score: {row['score']:>6.2f} | WR: {row['win_rate']:>5.2f}% | Sig: {row['total_signals']:>4} | Params: {json.dumps(p_dict)}")

        lines.extend(["", "  TOP 5 WORST IN-SAMPLE CONFIGURATIONS (with signals):", "  -------------------------------------------------------------------------"])
        for idx, row in df_worst.iterrows():
            p_dict = {k: row[k] for k in param_keys if k in row}
            lines.append(f"    Score: {row['score']:>6.2f} | WR: {row['win_rate']:>5.2f}% | Sig: {row['total_signals']:>4} | Params: {json.dumps(p_dict)}")

    return "\n".join(lines)


### MODIFIED ###: Function completely restructured for robust reporting
def generate_advanced_autoresearch_report(result_oos: PredictionResult, result_is: PredictionResult, best_params: dict, 
                                          df_oos: pd.DataFrame, df_is: pd.DataFrame, history: list, 
                                          param_space: dict, target_bars: int = 1) -> str:
    """
    Generates a comprehensive final meta-report comparing Out-of-Sample and In-Sample results.
    """
    w = 75
    # The most important result is Out-of-Sample, so it comes first.
    oos_report = generate_report(result_oos, df_oos, target_bars, title="OUT-OF-SAMPLE PERFORMANCE (VALIDATION)")
    
    # The In-Sample result is for comparison to check for overfitting.
    is_report = generate_report(result_is, df_is, target_bars, title="IN-SAMPLE PERFORMANCE (TRAINING)")
    
    meta_text = generate_meta_analysis_text(history, param_space, best_params)
    
    performance_drop = result_is.win_rate - result_oos.win_rate
    
    lines = [
        oos_report,
        "\n" + is_report,
        "",
        "  OVERFITTING ANALYSIS",
        f"    In-Sample Win Rate      : {result_is.win_rate * 100:.2f}%",
        f"    Out-of-Sample Win Rate  : {result_oos.win_rate * 100:.2f}%",
        f"    Performance Drop-off    : {performance_drop * 100:.2f}%",
        "    (A small drop-off indicates a more robust strategy)",
        "",
        "=" * w,
        " AUTORESEARCH META-ANALYSIS & CORRELATIONS (from In-Sample Data)",
        "=" * w,
        meta_text,
        "=" * w
    ]
    return "\n".join(lines)