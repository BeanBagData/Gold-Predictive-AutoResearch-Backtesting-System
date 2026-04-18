"""
backtest/performance.py
=======================
Classification and Predictive Accuracy reporting with Correlation Analysis.
"""
from __future__ import annotations
import json
import pandas as pd
from backtest.gold_backtester import PredictionResult


def generate_report(result: PredictionResult, df: pd.DataFrame, target_bars: int = 1,
                    title: str = "Strategy Report", bars_per_day: int = 288) -> str:
    """Basic backtest report for single runs."""
    w = 75
    if result.total_signals == 0:
        return f"{'='*w}\n {title.upper()}: NO SIGNALS GENERATED \n{'='*w}"

    days = len(df) / max(1, bars_per_day)
    signals_per_day = result.total_signals / max(1, days)
    target_min = target_bars * 5

    lines = [
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
        f"    Win-Rate Stability  : {result.win_rate_stability * 100:.1f}%",
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

    # Use the most recent 60% of valid experiments for correlations.
    # Older experiments come from different search-space regions and blend/invert the signal.
    recent_n = max(10, int(len(df_valid) * 0.60))
    df_corr = df_valid.iloc[-recent_n:]

    if len(df_corr) > 5 and df_corr['win_rate'].std() > 0:
        corrs = df_corr.corr(numeric_only=True)['win_rate'].drop(
            ['score', 'win_rate', 'total_signals', 'generation', 'win_rate_stability'],
            errors='ignore'
        )

        # Detect correlation sign reversals vs the older half (regime-jump indicator)
        older_n = len(df_valid) - recent_n
        sign_flips = set()
        if older_n >= 10:
            df_old = df_valid.iloc[:older_n]
            if df_old['win_rate'].std() > 0:
                old_corrs = df_old.corr(numeric_only=True)['win_rate'].drop(
                    ['score', 'win_rate', 'total_signals', 'generation', 'win_rate_stability'],
                    errors='ignore'
                )
                for param in corrs.index:
                    if param in old_corrs.index:
                        if corrs[param] * old_corrs[param] < 0 and abs(corrs[param]) > 0.1 and abs(old_corrs[param]) > 0.1:
                            sign_flips.add(param)

        lines.append(f"  (Recent-weighted: last {recent_n} of {len(df_valid)} experiments)")
        if sign_flips:
            lines.append(f"  ⚠ REGIME JUMP DETECTED — correlations reversed vs earlier runs: {', '.join(sorted(sign_flips))}")
        lines.append("  -------------------------------------------------------------------------")

        for k, c in corrs.sort_values(ascending=False).items():
            if pd.isna(c):
                continue
            strength = "Neutral"
            if abs(c) > 0.4:
                strength = "Strong"
            elif abs(c) > 0.2:
                strength = "Moderate"
            elif abs(c) > 0.1:
                strength = "Weak"
            direction = "Positive" if c > 0 else "Negative"
            label = f"({strength} {direction})" if strength != "Neutral" else "(Neutral)"
            flip_tag = " ⚠FLIP" if k in sign_flips else ""
            lines.append(f"    {k:<18} : {c:>6.2f}  {label}{flip_tag}")
    else:
        lines.append("    Not enough valid variance to calculate correlations.")

    lines.extend([
        "",
        "  PARAMETER SPACE EXPLORATION COVERAGE",
        "  -------------------------------------------------------------------------",
        "  [*] = experiments from old search regions exist outside current allowed bounds",
    ])

    for k, (lo, hi, _) in param_space.items():
        if k in df_hist.columns:
            t_min = df_hist[k].min()
            t_max = df_hist[k].max()
            coverage = (t_max - t_min) / max(1e-9, (hi - lo)) * 100

            warning = ""
            best_val = best_params.get(k, 0)
            if best_val <= lo + (hi - lo) * 0.05:
                warning = "<< AT MIN BOUND"
            elif best_val >= hi - (hi - lo) * 0.05:
                warning = ">> AT MAX BOUND"

            # Flag when cumulative history went outside the current allowed space
            hist_note = ""
            if t_min < lo - 0.5 or t_max > hi + 0.5:
                hist_note = " [*]"

            lines.append(
                f"    {k:<18} | Allowed: [{lo:g}, {hi:g}] | "
                f"Tested: [{t_min:.1f}, {t_max:.1f}] | "
                f"Cov: {coverage:3.0f}%  {warning}{hist_note}"
            )

    if len(df_valid) >= 10:
        df_best = df_valid.nlargest(5, 'score')
        df_worst = df_valid.nsmallest(5, 'score')
        param_keys = list(param_space.keys())

        lines.extend([
            "",
            "  TOP 5 BEST IN-SAMPLE CONFIGURATIONS:",
            "  -------------------------------------------------------------------------",
        ])
        for _, row in df_best.iterrows():
            p_dict = {k: row[k] for k in param_keys if k in row}
            lines.append(
                f"    Score: {row['score']:>6.2f} | WR: {row['win_rate']:>5.2f}% | "
                f"Sig: {row['total_signals']:>4} | Params: {json.dumps(p_dict)}"
            )

        lines.extend([
            "",
            "  TOP 5 WORST IN-SAMPLE CONFIGURATIONS (with signals):",
            "  -------------------------------------------------------------------------",
        ])
        for _, row in df_worst.iterrows():
            p_dict = {k: row[k] for k in param_keys if k in row}
            lines.append(
                f"    Score: {row['score']:>6.2f} | WR: {row['win_rate']:>5.2f}% | "
                f"Sig: {row['total_signals']:>4} | Params: {json.dumps(p_dict)}"
            )

    return "\n".join(lines)


def generate_advanced_autoresearch_report(
    result_oos: PredictionResult, result_is: PredictionResult,
    best_params: dict, df_oos: pd.DataFrame, df_is: pd.DataFrame,
    history: list, param_space: dict, target_bars: int = 1,
    bars_per_day: int = 288
) -> str:
    """
    Comprehensive final meta-report comparing Out-of-Sample and In-Sample results.
    """
    w = 75
    oos_report = generate_report(
        result_oos, df_oos, target_bars,
        title="OUT-OF-SAMPLE PERFORMANCE (VALIDATION)",
        bars_per_day=bars_per_day,
    )
    is_report = generate_report(
        result_is, df_is, target_bars,
        title="IN-SAMPLE PERFORMANCE (TRAINING)",
        bars_per_day=bars_per_day,
    )

    meta_text = generate_meta_analysis_text(history, param_space, best_params)

    performance_drop = result_is.win_rate - result_oos.win_rate
    oos_better = result_oos.win_rate > result_is.win_rate

    lines = [
        oos_report,
        "\n" + is_report,
        "",
        "  OVERFITTING ANALYSIS",
        f"    In-Sample Win Rate      : {result_is.win_rate * 100:.2f}%",
        f"    Out-of-Sample Win Rate  : {result_oos.win_rate * 100:.2f}%",
        f"    Performance Drop-off    : {performance_drop * 100:.2f}%",
    ]

    if oos_better:
        lines.append(
            "    ⚠ OOS OUTPERFORMS IS: The OOS period may have unusually favourable"
        )
        lines.append(
            "      conditions for these indicators. Monitor live performance carefully."
        )
    else:
        lines.append("    (A small drop-off indicates a more robust strategy)")

    lines.extend([
        "",
        "=" * w,
        " AUTORESEARCH META-ANALYSIS & CORRELATIONS (from In-Sample Data)",
        "=" * w,
        meta_text,
        "=" * w,
    ])
    return "\n".join(lines)
