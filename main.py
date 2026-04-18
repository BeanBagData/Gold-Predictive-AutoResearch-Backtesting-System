"""
Claude Gold — Offline Predictor & AutoResearch (TA Predictive v2.0)
====================================================================
Hierarchical parameter optimisation for predictive direction.

AUTOMATIC DATA LOADING:
  The script automatically scans the 'data/' directory for a CSV file.
  Ensure exactly ONE OHLCV CSV is in the 'data/' folder before running.

COMMANDS:
  1. DEBUG: Run a baseline backtest on the current CSV to verify data loading.
     python main.py --mode debug --horizon 15

  2. MANUAL: Run a backtest using specific parameters from a saved JSON report.
     python main.py --mode manual --params output/best_report.json --horizon 15

  3. AUTO (Fresh): Start a new hierarchical search session.
     python main.py --mode auto --n 50 --horizon 15

  4. AUTO (Resume): Continue a previous search using 'best_report.json' as a seed.
     python main.py --mode auto --n 100 --resume --horizon 10
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.loader import load_mt5_csv
from backtest.gold_backtester import run_predictive_backtest, PredictorParams
from backtest.performance import generate_report, generate_advanced_autoresearch_report
from core.gold_autoresearch import GoldAutoResearch, INITIAL_PARAM_SPACE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-7s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

def _find_csv() -> str:
    EXCLUDE_PATTERNS = ['log', 'result', 'state', 'params', 'autoresearch', 'claude_gold', 'best_report', 'top_worst_pools']
    candidates = glob.glob('data/*.csv')
    filtered = [c for c in candidates if not any(p in os.path.basename(c).lower() for p in EXCLUDE_PATTERNS)]
    if len(filtered) == 1:
        return filtered[0]
    if len(filtered) == 0:
        raise FileNotFoundError("No CSV found in 'data/'. Place your MT5 export there.")
    raise FileNotFoundError(f"Multiple CSVs in 'data/': {[os.path.basename(c) for c in filtered]}. Leave only one.")

def _load_params_json(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    return data.get('params', data.get('best_params', data))

def _load_resume_state(path: str) -> tuple:
    """Returns (params_dict, best_score, param_space, history) from saved best_report.json."""
    with open(path) as f:
        data = json.load(f)
    params = data.get('params', data.get('best_params', {}))
    best_score = float(data.get('best_oos_score', 0.0))
    raw_space = data.get('current_search_space', {})
    param_space = {k: tuple(v) for k, v in raw_space.items()
                   if isinstance(v, (list, tuple)) and len(v) == 3}
    
    history = data.get('history', [])
    return params, best_score, param_space or None, history

def _dict_to_params(d: dict) -> PredictorParams:
    p = PredictorParams()
    for k, v in d.items():
        if hasattr(p, k):
            setattr(p, k, type(getattr(p, k))(v))
    return p

def _save_report(text: str, name: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    path = os.path.join(out_dir, f'{name}_{ts}.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    logger.info("Saved: %s", path)
    return path

def _detect_bars_per_day(df: pd.DataFrame) -> int:
    median_minutes = df.index.to_series().diff().median().total_seconds() / 60
    bars = int(round(1440 / max(1, median_minutes)))
    logger.info("Auto-detected bar interval: %.1f min → %d bars/day", median_minutes, bars)
    return bars

def run_manual(args):
    csv_path = args.csv if args.csv else _find_csv()
    df = load_mt5_csv(csv_path)
    target_bars = max(1, args.horizon)
    bars_per_day = _detect_bars_per_day(df)
    logger.info("Manual Mode: Loading %s (Horizon: %d bars)", csv_path, target_bars)

    params_dict = _load_params_json(args.params) if (args.params and os.path.exists(args.params)) else {}
    ea_params = _dict_to_params(params_dict)

    result = run_predictive_backtest(df, params=ea_params, target_bars=target_bars)
    report = generate_report(result, df, target_bars=target_bars, bars_per_day=bars_per_day)
    print(report)
    _save_report(report, 'ta_predictive_backtest', args.out)

def run_auto(args):
    csv_path = args.csv if args.csv else _find_csv()
    df_full = load_mt5_csv(csv_path)
    
    split_index = int(len(df_full) * 0.8)
    if split_index < 200 or len(df_full) - split_index < 50:
        raise ValueError("Dataset is too small to perform a meaningful IS/OOS split.")
        
    df_insample = df_full.iloc[:split_index]
    df_oos = df_full.iloc[split_index:]

    target_bars = max(1, args.horizon)

    initial_params = None
    initial_score = 0.0
    initial_space = None
    initial_history = []

    if args.params and os.path.exists(args.params):
        initial_params, initial_score, initial_space, initial_history = _load_resume_state(args.params)
        if initial_score != 0:
            logger.info("Resuming with OOS_score=%.4f and %d historical runs loaded.", initial_score, len(initial_history))

    bars_per_day = _detect_bars_per_day(df_full)
    logger.info("AutoResearch: %d cycles on %s (Horizon: %d bars)", args.n, csv_path, target_bars)

    ar = GoldAutoResearch(
        df_insample=df_insample,
        df_oos=df_oos,
        seed=args.seed,
        output_dir=args.out,
        initial_params=initial_params,
        initial_score=initial_score,
        initial_space=initial_space,
        initial_history=initial_history,
        target_bars=target_bars,
        bars_per_day=bars_per_day,
    )
    history = ar.run(n_ollama_cycles=args.n, verbose=not args.quiet)

    if ar.best_params and ar.best_score > -1000.0:
        ea_best = _dict_to_params(ar.best_params)
        
        res_insample = run_predictive_backtest(df_insample, params=ea_best, target_bars=target_bars)
        res_oos = run_predictive_backtest(df_oos, params=ea_best, target_bars=target_bars)
        
        report_text = generate_advanced_autoresearch_report(
            result_oos=res_oos, result_is=res_insample, best_params=ar.best_params,
            df_oos=df_oos, df_is=df_insample, history=history,
            param_space=ar.param_space, target_bars=target_bars,
            bars_per_day=bars_per_day,
        )
    else:
        report_text = "No configurations generated enough signals."

    print(report_text)
    _save_report(report_text, 'ta_autoresearch_report', args.out)

def run_debug(args):
    csv_path = args.csv if args.csv else _find_csv()
    df = load_mt5_csv(csv_path)
    target_bars = max(1, args.horizon)
    bars_per_day = _detect_bars_per_day(df)
    logger.info("Debug Mode: Scanning %s (Horizon: %d bars)", csv_path, target_bars)

    p = PredictorParams()
    result = run_predictive_backtest(df, params=p, target_bars=target_bars)
    print(generate_report(result, df, target_bars=target_bars, bars_per_day=bars_per_day))

def main():
    p = argparse.ArgumentParser(description='Claude Gold TA Predictor & AutoResearch')
    p.add_argument('--mode', default='debug', choices=['manual', 'auto', 'debug'])
    p.add_argument('--seed', default=42, type=int)
    p.add_argument('--n', default=30, type=int)
    p.add_argument('--params', default=None)
    p.add_argument('--out', default='output')
    p.add_argument('--quiet', action='store_true')
    p.add_argument('--resume', action='store_true', help='Resume from best_report.json')
    p.add_argument('--horizon', default=3, type=int, help='Target bars ahead to predict')
    p.add_argument('--csv', default=None, help='Direct path to CSV (pipeline use only)')
    args = p.parse_args()

    if args.resume:
        if args.params is None:
            resume_path = os.path.join(args.out, 'best_report.json')
            # Fallback to local directory if not in output folder
            if not os.path.exists(resume_path) and os.path.exists('best_report.json'):
                resume_path = 'best_report.json'
                
            if os.path.exists(resume_path):
                args.params = resume_path
                logger.info("Auto-Resuming from: %s", resume_path)
            else:
                logger.warning("Resume flag passed, but 'best_report.json' not found.")

    dispatch = {'manual': run_manual, 'auto': run_auto, 'debug': run_debug}
    dispatch[args.mode](args)

if __name__ == '__main__':
    main()
