"""
core/gold_autoresearch.py
==========================
v2.6: Universal Discovery Mode
Reverted to a broad search to find the optimal M1 and M5 parameters
with the new v2.5.1 backtesting logic. Only hard-coded RSI is locked.
"""
from __future__ import annotations

import copy
import json
import logging
import math
import os
import random
import time
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

from backtest.gold_backtester import run_predictive_backtest, PredictorParams
from backtest.performance import generate_meta_analysis_text
from .ollama_advisor import OllamaAdvisor

logger = logging.getLogger(__name__)

# v2.6 LOCKED: Only the absolute hard-coded EA logic is frozen.
STATIC_EA_PARAMS = {
    'rsi_period':      5,
    'rsi_oversold':    29.0,
    'rsi_overbought':  79.0,
}

# v2.6 RESEARCHABLE: Wide search space for full discovery on M1/M5 data.
INITIAL_PARAM_SPACE = {
    'ema_fast':     (3, 50, 1),
    'ema_slow':     (20, 200, 1),
    'macd_fast':    (5, 25, 1),
    'macd_slow':    (25, 80, 1),
    'macd_signal':  (2, 15, 1),
    'adx_period':   (5, 20, 1),
    'adx_thresh':   (15.0, 60.0, 1.0),  # Wide range to re-discover the "Ignition" point
    'stoch_k':      (3, 20, 1),
    'stoch_d':      (2, 5, 1),
    'stoch_smooth': (1, 15, 1),
    'bb_period':    (10, 60, 1),
    'bb_stdev':     (1.0, 3.0, 0.1),
    'atr_period':   (2, 30, 1),
    'atr_mult':     (0.5, 3.0, 0.1),
}

INT_PARAMS = {
    'ema_fast', 'ema_slow', 'macd_fast', 'macd_slow',
    'macd_signal', 'adx_period', 'stoch_k', 'stoch_d', 'stoch_smooth',
    'bb_period', 'atr_period'
}

# Standard Hierarchical Phasing for Discovery
PHASE_1_KEYS = ['ema_fast', 'ema_slow', 'adx_thresh', 'adx_period', 'macd_fast', 'macd_slow']
PHASE_2_KEYS = ['stoch_k', 'stoch_d', 'stoch_smooth', 'macd_signal']
PHASE_3_KEYS = ['bb_period', 'bb_stdev', 'atr_period', 'atr_mult']

EXPERIMENTS_PER_OLLAMA_CYCLE = 15


def _clip(val: float, lo: float, hi: float, is_int: bool = False) -> float:
    val = max(lo, min(hi, val))
    return round(val) if is_int else val


def _sanitize_params(params: dict) -> dict:
    params.update(STATIC_EA_PARAMS)
    for k in INT_PARAMS:
        if k in params and params[k] < 1: params[k] = 1
    if 'ema_fast' in params and 'ema_slow' in params:
        params['ema_fast'] = max(3, params['ema_fast'])
        if params['ema_fast'] >= params['ema_slow']:
            params['ema_fast'] = max(3, params['ema_slow'] - 2)
    if 'macd_fast' in params and 'macd_slow' in params:
        params['macd_fast'] = max(2, params['macd_fast'])
        if params['macd_fast'] >= params['macd_slow']:
            params['macd_fast'] = max(2, params['macd_slow'] - 5)
    return params


def compute_predictive_score(result, n_bars: int, bars_per_day: int = 288) -> dict:
    wr = result.win_rate
    n = result.total_signals
    days = max(1.0, n_bars / bars_per_day)
    signals_per_day = n / days

    breakdown = {
        'win_rate': round(wr * 100, 2),
        'total_signals': n,
        'signals_per_day': round(signals_per_day, 1),
        'up_win_rate': round(result.up_win_rate * 100, 2),
        'down_win_rate': round(result.down_win_rate * 100, 2),
        'consec_correct': result.max_consec_correct,
        'consec_incorrect': result.max_consec_incorrect,
        'win_rate_stability': round(result.win_rate_stability * 100, 2)
    }

    if n == 0:
        breakdown['score'] = -1000.0
        return {'score': -1000.0, 'breakdown': breakdown}

    accuracy_score = (wr - 0.50) * 100

    # Scalping signal floor: demand high entry frequency relative to bars available.
    # M1 (1440/day) targets 60/day — one signal per ~24 min.
    # M5 (288/day) targets 30/day — one signal per ~48 min.
    # Both are calibrated for active scalping, not swing-style low-frequency filters.
    signals_per_day_target = 60.0 if bars_per_day > 300 else 30.0
    expected_minimum_signals = max(300, int(days * signals_per_day_target))

    volume_factor = 1.0
    if n < expected_minimum_signals:
        volume_factor = (n / expected_minimum_signals) ** 3

    freq_mod = math.log10(max(1, signals_per_day)) if signals_per_day >= 2.0 else 0.5

    if accuracy_score > 0:
        imbalance = abs(result.up_signals - result.down_signals) / max(1, n)
        balance_penalty = 1.0 - (imbalance * 0.5)
        stability_bonus = 1.0 + (result.win_rate_stability * 0.15)
        score = accuracy_score * freq_mod * balance_penalty * stability_bonus * volume_factor
    else:
        score = accuracy_score

    breakdown['score'] = round(score, 4)
    return {'score': score, 'breakdown': breakdown}


class GoldAutoResearch:
    def __init__(self, df_insample: pd.DataFrame, df_oos: pd.DataFrame, seed: int = 42,
                 output_dir: str = '.', initial_params: dict = None,
                 initial_score: float = 0.0, initial_space: dict = None, initial_history: list = None,
                 target_bars: int = 1, bars_per_day: int = 288):

        self.df_insample = df_insample
        self.df_oos = df_oos
        self.rng = random.Random(seed)
        self.output_dir = output_dir
        self.target_bars = target_bars
        self.bars_per_day = bars_per_day
        os.makedirs(self.output_dir, exist_ok=True)

        self.best_score = initial_score if initial_params else -99999.0
        self.best_params = initial_params
        self.best_breakdown = {}
        self.history = initial_history if initial_history else []
        self._best_report_path = os.path.join(self.output_dir, 'best_report.json')

        self.param_space = copy.deepcopy(initial_space) if initial_space else copy.deepcopy(INITIAL_PARAM_SPACE)

        # Generic, well-rounded baseline for starting a fresh discovery
        self.baseline_params = {
            'ema_fast': 12, 'ema_slow': 50,
            'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
            'adx_period': 14, 'adx_thresh': 25.0,
            'stoch_k': 14, 'stoch_d': 3, 'stoch_smooth': 3,
            'bb_period': 20, 'bb_stdev': 2.0,
            'atr_period': 14, 'atr_mult': 2.0,
            **STATIC_EA_PARAMS
        }
        self.ollama_advisor = OllamaAdvisor()

    def _dict_to_params(self, d: dict) -> PredictorParams:
        p = PredictorParams()
        for k, v in d.items():
            if hasattr(p, k):
                setattr(p, k, type(getattr(p, k))(v))
        return p

    def _mutate_hierarchical(self, base: dict, generation: int) -> dict:
        mutated = dict(base)
        active_keys = list(PHASE_1_KEYS)
        if generation > 30: active_keys.extend(PHASE_2_KEYS)
        if generation > 60: active_keys.extend(PHASE_3_KEYS)

        researchable_keys = [k for k in active_keys if k in self.param_space]
        for k in researchable_keys:
            lo, hi, step = self.param_space[k]
            if self.rng.random() < 0.30:
                noise_std = max((hi - lo) * 0.15, step)
                mutated[k] = _clip(base.get(k, (lo + hi) / 2) + self.rng.gauss(0, noise_std), lo, hi, k in INT_PARAMS)
        return _sanitize_params(mutated)

    def _save_best(self, params, score_oos, breakdown_oos, gen, score_is, breakdown_is):
        self.best_score = score_oos
        self.best_params = dict(params)
        self.best_breakdown = breakdown_oos

        valid_hist = [h for h in self.history if h.get('total_signals', 0) > 0]
        top_5_best = sorted(valid_hist, key=lambda x: x.get('score', -999.0), reverse=True)[:5]
        top_5_worst = sorted(valid_hist, key=lambda x: x.get('score', 999.0))[:5]

        report = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "generation": gen,
            "best_oos_score": round(score_oos, 6),
            "oos_metrics": breakdown_oos,
            "corresponding_is_score": round(score_is, 6),
            "is_metrics": breakdown_is,
            "params": params,
            "current_search_space": self.param_space,
            "top_5_best": top_5_best,
            "top_5_worst": top_5_worst,
            "history": self.history
        }
        with open(self._best_report_path, 'w') as f:
            json.dump(report, f, indent=2)

    def run(self, n_ollama_cycles: int = 10, verbose: bool = True) -> list:
        if verbose:
            print(f"\n{'='*60}")
            print(f" TA Predictive AutoResearch (v2.6: DISCOVERY MODE)")
            print(f" Target: Find best parameters for {self.target_bars} bars ahead")
            print(f"{'='*60}")

        generation = 0
        if self.history:
            generation = max([h.get('generation', 0) for h in self.history]) + 1

        for cycle in range(n_ollama_cycles):
            phase = 1 if generation <= 30 else (2 if generation <= 60 else 3)
            if verbose:
                print(f"\n--- Cycle {cycle+1}/{n_ollama_cycles} (Phase {phase}: Gens {generation}-{generation+EXPERIMENTS_PER_OLLAMA_CYCLE}) ---")

            for _ in range(EXPERIMENTS_PER_OLLAMA_CYCLE):
                if self.best_params is None:
                    candidate = self.baseline_params if generation == 0 else self._mutate_hierarchical(self.baseline_params, generation)
                else:
                    candidate = self._mutate_hierarchical(self.best_params, generation)

                candidate.update(STATIC_EA_PARAMS)

                res_is = run_predictive_backtest(self.df_insample, self._dict_to_params(candidate), self.target_bars)
                sc_is = compute_predictive_score(res_is, len(self.df_insample), self.bars_per_day)

                b_is = sc_is['breakdown']
                self.history.append({
                    'generation': generation, 'score': sc_is['score'],
                    'win_rate': b_is.get('win_rate', 0.0), 'total_signals': b_is.get('total_signals', 0),
                    'win_rate_stability': b_is.get('win_rate_stability', 0.0), **candidate
                })

                if b_is.get('total_signals', 0) > 0:
                    res_oos = run_predictive_backtest(self.df_oos, self._dict_to_params(candidate), self.target_bars)
                    sc_oos = compute_predictive_score(res_oos, len(self.df_oos), self.bars_per_day)

                    if sc_oos['score'] > self.best_score:
                        self._save_best(candidate, sc_oos['score'], sc_oos['breakdown'], generation, sc_is['score'], b_is)
                        if verbose:
                            print(f"  *** New Best! OOS Acc: {sc_oos['breakdown'].get('win_rate'):.2f}% | "
                                  f"OOS Signals: {sc_oos['breakdown'].get('total_signals', 0)} | "
                                  f"Score: {sc_oos['score']:.2f}")

                generation += 1

            if self.best_params:
                if verbose: print(f"  [Ollama Advisor] Analyzing cycle and suggesting mutation...")
                valid_hist = [h for h in self.history if h.get('total_signals', 0) > 0]
                top_best = sorted(valid_hist, key=lambda x: x.get('score', -999.0), reverse=True)[:5]
                top_worst = sorted(valid_hist, key=lambda x: x.get('score', 999.0))[:5]
                suggested = self.ollama_advisor.suggest_mutation(
                    current_params=self.best_params, metrics=self.best_breakdown,
                    param_space=self.param_space, iteration=cycle + 1,
                    stale_count=0, target_bars=self.target_bars,
                    top_best=top_best, top_worst=top_worst
                )
                if suggested:
                    suggested = _sanitize_params(suggested)
                    generation += 1

            if (cycle + 1) % 3 == 0 and self.best_params and len(self.history) >= 45:
                meta_report_text = generate_meta_analysis_text(self.history, self.param_space, self.best_params)
                new_bounds = self.ollama_advisor.adjust_search_space(self.param_space, meta_report_text)
                if new_bounds: self.param_space.update(new_bounds)

        return self.history
