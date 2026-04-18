"""
pipeline.py
===========
v1.0: Meta-AutoResearch Pipeline

Scans data/ for all CSVs, auto-detects their timeframe by bar interval,
then runs sequential autoresearch phases (longest history first = M5 before M1).

For each phase it dynamically adjusts --n to converge on the known sweet spot,
using Ollama analysis + algorithmic fallback. Stops after 3 consecutive sweet
spot hits or MAX_ATTEMPTS total, whichever comes first.

Usage:
    python pipeline.py                        # auto-detect everything
    python pipeline.py --start-n 10          # override starting N
    python pipeline.py --max-attempts 20     # limit total attempts per phase
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.ollama_advisor import OllamaAdvisor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-7s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# ─── Sweet Spot Benchmarks (from research history) ───────────────────────────
# These define what we're hunting for. Any result inside these ranges is a win.
SWEET_SPOTS = {
    'm5': {
        'label':        'M5 Institutional Flow',
        'horizon':      3,        # 3 × 5 min = 15 min prediction
        'wr_min':       53.5,
        'wr_max':       55.0,
        'spd_min':      25.0,
        'spd_max':      45.0,
        'benchmark_wr': 54.38,
        'benchmark_spd': 42.1,
    },
    'm1': {
        'label':        'M1 Momentum Ignition',
        'horizon':      15,       # 15 × 1 min = 15 min prediction
        'wr_min':       51.5,
        'wr_max':       52.5,
        'spd_min':      80.0,
        'spd_max':      120.0,
        'benchmark_wr': 52.0,
        'benchmark_spd': 116.0,
    },
}

# N profiles are phase-order based, not timeframe based.
# Phase 0 = first CSV (longest history, fresh discovery) → needs broad N range.
# Phase 1 = second CSV (seeded from phase 0 best params) → needs far fewer cycles.
PHASE_N_PROFILES = [
    {'start_n': 10, 'max_n': 150, 'n_step': 10},  # Phase 0: first CSV  — broad discovery
    {'start_n':  5, 'max_n':  25, 'n_step':  5},  # Phase 1: second CSV — seeded, fast
]

MIN_N             = 3
CONVERGENCE_HITS  = 5   # must hit sweet spot 5 times before finalising
MAX_ATTEMPTS      = 25  # raised to accommodate finer step=1 convergence passes
EXCLUDE_PATTERNS  = ['log', 'result', 'state', 'params', 'autoresearch',
                     'claude_gold', 'best_report', 'top_worst_pools']


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class RunResult:
    n: int
    wr: float
    spd: float
    stability: float
    total_signals: int
    oos_score: float
    verdict: str
    in_sweet_spot: bool = False
    # Extended quality metrics
    up_wr: float = 0.0
    down_wr: float = 0.0
    consec_correct: int = 0
    consec_incorrect: int = 0


@dataclass
class PhaseState:
    tf: str
    csv_path: str
    out_dir: str
    n_profile: dict      # start_n, max_n, n_step for this phase
    history: list[RunResult] = field(default_factory=list)
    best: Optional[RunResult] = None
    converge_count: int = 0
    attempt: int = 0


# ─── CSV Detection ───────────────────────────────────────────────────────────

def detect_timeframe(path: str) -> tuple[str, int]:
    """
    Identify timeframe and bars_per_day for a CSV file.

    Strategy:
      1. Check filename for _M1_, _M5_, _M15_, etc. (MT5 standard export naming).
         This is fast, reliable, and works when both files are in data/ together.
      2. If no match in filename, fall back to reading ~100 rows and measuring
         the median bar interval from the actual timestamps.
    """
    fname = os.path.basename(path).upper()

    # Known MT5 timeframe tokens → (tf_key, bars_per_day)
    TF_MAP = [
        ('_M1_',   'm1',   1440),
        ('_M5_',   'm5',   288),
        ('_M15_',  'm15',  96),
        ('_M30_',  'm30',  48),
        ('_H1_',   'h1',   24),
        ('_H4_',   'h4',   6),
        ('_D1_',   'd1',   1),
    ]
    for token, tf, bpd in TF_MAP:
        if token in fname:
            logger.info("Timeframe identified from filename: %s → %s (%d bars/day)",
                        os.path.basename(path), tf.upper(), bpd)
            return tf, bpd

    # Fallback: detect from actual bar intervals
    logger.info("No timeframe token in filename — detecting from bar intervals...")
    with open(path, 'r') as f:
        first = f.readline()
    sep = '\t' if '\t' in first else ','

    df = pd.read_csv(path, sep=sep, nrows=100)
    df.columns = [c.strip('<>').lower() for c in df.columns]

    if 'date' in df.columns and 'time' in df.columns:
        df['dt'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    elif 'datetime' in df.columns:
        df['dt'] = pd.to_datetime(df['datetime'])
    else:
        raise ValueError(f"Cannot parse datetime from {path}")

    df = df.sort_values('dt')
    median_min = df['dt'].diff().median().total_seconds() / 60
    bars_per_day = int(round(1440 / max(0.1, median_min)))

    if median_min <= 1.5:
        tf = 'm1'
    elif median_min <= 6:
        tf = 'm5'
    elif median_min <= 16:
        tf = 'm15'
    else:
        tf = f'm{int(round(median_min))}'

    logger.info("Timeframe detected from intervals: %.1f min/bar → %s (%d bars/day)",
                median_min, tf.upper(), bars_per_day)
    return tf, bars_per_day


def scan_csvs() -> list[dict]:
    """Find all valid CSVs in data/, detect their timeframes, sort longest first."""
    results = []
    if not os.path.isdir('data'):
        raise FileNotFoundError("'data/' directory not found.")

    for fname in os.listdir('data'):
        if not fname.endswith('.csv'):
            continue
        if any(p in fname.lower() for p in EXCLUDE_PATTERNS):
            continue
        path = os.path.join('data', fname)
        try:
            tf, bpd = detect_timeframe(path)
            with open(path) as fh:
                rows = sum(1 for _ in fh) - 1
            days = round(rows / max(1, bpd), 1)
            results.append({'path': path, 'tf': tf, 'bars_per_day': bpd,
                            'rows': rows, 'days': days, 'fname': fname})
            logger.info("Detected: %s → %s, ~%.0f days (%d bars)", fname, tf.upper(), days, rows)
        except Exception as e:
            logger.warning("Skipping %s: %s", fname, e)

    # Longest history first — ensures M5 (17 months) runs before M1 (3 months)
    results.sort(key=lambda x: x['days'], reverse=True)
    return results


# ─── Result Evaluation ───────────────────────────────────────────────────────

def evaluate(report: dict, tf: str, n: int) -> RunResult:
    """Score a best_report.json against sweet spot targets including quality gates."""
    target = SWEET_SPOTS.get(tf, SWEET_SPOTS['m5'])
    m = report.get('oos_metrics', {})

    wr               = float(m.get('win_rate', 0))
    spd              = float(m.get('signals_per_day', 0))
    stability        = float(m.get('win_rate_stability', 0))
    total_sigs       = int(m.get('total_signals', 0))
    oos_score        = float(report.get('best_oos_score', 0))
    up_wr            = float(m.get('up_win_rate', 0))
    down_wr          = float(m.get('down_win_rate', 0))
    consec_correct   = int(m.get('consec_correct', 0))
    consec_incorrect = int(m.get('consec_incorrect', 0))

    # ── Core range checks ─────────────────────────────────────────────────────
    in_wr  = target['wr_min'] <= wr <= target['wr_max']
    in_spd = target['spd_min'] <= spd <= target['spd_max']

    # ── Quality gates (all must pass for a true sweet spot) ──────────────────
    # Stability must exceed win rate — consistency outweighs raw accuracy.
    # A result with WR=54% and stability=55% passes; WR=54% and stability=53% fails.
    # No hardcoded floor: the relationship to WR is the meaningful signal.
    stability_beats_wr = stability > wr
    # Both directions must be predictive — no one-sided bias hiding in the WR
    both_dirs_above_51 = up_wr > 51.0 and down_wr > 51.0
    # More consecutive correct than incorrect — streaks go the right way
    streaks_healthy = consec_correct > consec_incorrect

    quality_pass = stability_beats_wr and both_dirs_above_51 and streaks_healthy
    sweet = in_wr and in_spd and quality_pass

    # ── Verdict ───────────────────────────────────────────────────────────────
    if sweet:
        verdict = 'SWEET_SPOT'
    elif in_wr and in_spd and not quality_pass:
        verdict = 'QUALITY_FAIL'      # range correct but quality gates failed
    elif spd > target['spd_max']:
        verdict = 'TOO_FREQUENT'      # signals far above target — params too loose
    elif wr > target['wr_max'] and spd < target['spd_min']:
        verdict = 'OVERFITTING'       # ghost hunter — high WR, too few trades
    elif wr > target['wr_max']:
        verdict = 'OVER_OPTIMIZED'
    elif in_wr and spd < target['spd_min']:
        verdict = 'TOO_RARE'          # right accuracy, not enough signals
    elif wr >= target['wr_min'] - 1.5 or spd >= target['spd_min'] * 0.7:
        verdict = 'APPROACHING'
    else:
        verdict = 'UNDERFITTING'

    return RunResult(
        n=n, wr=wr, spd=spd, stability=stability,
        total_signals=total_sigs, oos_score=oos_score,
        verdict=verdict, in_sweet_spot=sweet,
        up_wr=up_wr, down_wr=down_wr,
        consec_correct=consec_correct, consec_incorrect=consec_incorrect,
    )


# ─── N Adjustment ────────────────────────────────────────────────────────────

def dynamic_step(result: RunResult, target: dict, n_profile: dict) -> int:
    """
    Step size collapses to 1 once we're inside or approaching the sweet spot range.
    This allows Ollama to make fine single-cycle comparisons to pick the best N.
    Outside the range we use the profile step for efficient traversal.
    """
    wr_close = result.wr >= (target['wr_min'] - 1.5)
    spd_close = result.spd >= (target['spd_min'] * 0.75)
    # TOO_FREQUENT excluded — signals 2× target need large N jumps to explore new space, not fine-step
    near = result.verdict in ('SWEET_SPOT', 'APPROACHING', 'QUALITY_FAIL') or (wr_close and spd_close)
    return 1 if near else n_profile['n_step']


def algorithmic_n(state: PhaseState, result: RunResult) -> int:
    """Pure numeric fallback for N-adjustment when Ollama is unavailable."""
    target  = SWEET_SPOTS.get(state.tf, SWEET_SPOTS['m5'])
    max_n   = state.n_profile['max_n']
    step    = dynamic_step(result, target, state.n_profile)
    h       = state.history
    n       = result.n

    if result.verdict in ('SWEET_SPOT', 'QUALITY_FAIL'):
        if len(h) >= 2 and h[-2].oos_score > result.oos_score:
            return max(MIN_N, h[-2].n)        # backtrack if score dropped
        return min(n + step, max_n)           # step=1 here — fine-grained search

    elif result.verdict == 'UNDERFITTING':
        return min(n + step * 2, max_n)       # large jumps far from target

    elif result.verdict == 'APPROACHING':
        return min(n + step, max_n)           # step=1 once close

    elif result.verdict == 'TOO_FREQUENT':
        # Signals too high — N isn't the real lever, but try a divergent jump to
        # force Ollama into different param regions (higher ADX thresh, higher bb_stdev)
        good = [r for r in h if r.spd <= target['spd_max']]
        if good:
            return max(good, key=lambda r: r.oos_score).n
        # No prior run had acceptable signal count — try a larger N jump to escape
        return min(n + step * 3, max_n)

    elif result.verdict == 'TOO_RARE':
        good = [r for r in h if r.spd >= target['spd_min']]
        if good:
            return max(good, key=lambda r: r.oos_score).n
        return max(MIN_N, n - step)

    elif result.verdict in ('OVERFITTING', 'OVER_OPTIMIZED'):
        good = [r for r in h if r.spd >= target['spd_min']]
        if good:
            best_good_n = max(good, key=lambda r: r.oos_score).n
            return max(MIN_N, (best_good_n + n) // 2)
        return max(MIN_N, n - step)

    return min(n + step, max_n)


def choose_next_n(state: PhaseState, result: RunResult,
                  advisor: OllamaAdvisor) -> int:
    """Try Ollama first; fall back to algorithmic if it fails or returns nothing."""
    target  = SWEET_SPOTS.get(state.tf, SWEET_SPOTS['m5'])
    max_n   = state.n_profile['max_n']
    hist_dicts = [{'n': r.n, 'wr': r.wr, 'spd': r.spd, 'verdict': r.verdict}
                  for r in state.history]

    # Compute delta vs previous run so Ollama can detect stagnation
    prev = state.history[-2] if len(state.history) >= 2 else None
    delta_wr    = round(result.wr    - prev.wr,        3) if prev else 0.0
    delta_score = round(result.oos_score - prev.oos_score, 4) if prev else 0.0

    metrics = {
        'win_rate':          result.wr,
        'signals_per_day':   result.spd,
        'win_rate_stability': result.stability,
        'delta_wr':          delta_wr,
        'delta_oos_score':   delta_score,
    }

    suggestion = advisor.suggest_n_adjustment(
        current_n=result.n, metrics=metrics, sweet_spot=target,
        history=hist_dicts, verdict=result.verdict
    )
    if suggestion and 'suggested_n' in suggestion:
        n_new = int(suggestion['suggested_n'])
        reasoning = suggestion.get('reasoning', '')
        n_new = max(MIN_N, min(max_n, n_new))   # hard clamp — Ollama cannot exceed ceiling
        logger.info("Ollama N suggestion: %d (ceiling=%d) — %s", n_new, max_n, reasoning)
        return n_new

    n_new = algorithmic_n(state, result)
    logger.info("Algorithmic N suggestion: %d (verdict=%s, ceiling=%d)",
                n_new, result.verdict, max_n)
    return n_new


# ─── Subprocess Runner ───────────────────────────────────────────────────────

def run_main(csv_path: str, n: int, horizon: int, out_dir: str,
             params_path: Optional[str] = None) -> bool:
    cmd = [
        sys.executable, 'main.py',
        '--mode', 'auto',
        '--csv',  csv_path,
        '--horizon', str(horizon),
        '--n',    str(n),
        '--out',  out_dir,
    ]
    if params_path and os.path.exists(params_path):
        cmd += ['--params', params_path]

    logger.info("CMD: %s", ' '.join(cmd))
    try:
        result = subprocess.run(cmd, timeout=7200)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        logger.error("Run timed out (2h). Continuing with last saved report.")
        return False
    except Exception as e:
        logger.error("Subprocess error: %s", e)
        return False


def read_report(out_dir: str) -> Optional[dict]:
    path = os.path.join(out_dir, 'best_report.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ─── Display ─────────────────────────────────────────────────────────────────

def _gate(val: bool) -> str:
    return '✓' if val else '✗'

def print_result(state: PhaseState, result: RunResult):
    target  = SWEET_SPOTS.get(state.tf, SWEET_SPOTS['m5'])
    sw      = '✓ SWEET SPOT' if result.in_sweet_spot else f'  {result.verdict}'
    step    = dynamic_step(result, target, state.n_profile)

    stab_beats_wr  = result.stability > result.wr
    both_dirs      = result.up_wr > 51.0 and result.down_wr > 51.0
    streaks_ok     = result.consec_correct > result.consec_incorrect

    print(f"""
{'─'*62}
  {state.tf.upper()} | Attempt {state.attempt}/{MAX_ATTEMPTS} | N={result.n} | step={step} | {sw}
{'─'*62}
  Win Rate:        {result.wr:.2f}%   target {target['wr_min']}–{target['wr_max']}%
  Signals/Day:     {result.spd:.1f}    target {target['spd_min']}–{target['spd_max']}{'  !! TOO HIGH' if result.spd > target['spd_max'] else ''}
  Total Signals:   {result.total_signals}
  OOS Score:       {result.oos_score:.4f}
{'─'*62}
  Quality Gates:
    Stability:     {result.stability:.1f}%
    Stab > WR:     {result.stability:.1f}% vs {result.wr:.2f}%  {_gate(stab_beats_wr)}
    Up WR:         {result.up_wr:.2f}%  {_gate(result.up_wr > 51.0)} (need >51%)
    Down WR:       {result.down_wr:.2f}%  {_gate(result.down_wr > 51.0)} (need >51%)
    Streaks:       correct={result.consec_correct} incorrect={result.consec_incorrect}  {_gate(streaks_ok)}
  Converge:        {state.converge_count}/{CONVERGENCE_HITS}
{'─'*62}""")


# ─── Phase Runner ────────────────────────────────────────────────────────────

def run_phase(csv_info: dict, seed_params: Optional[str], advisor: OllamaAdvisor,
              phase_index: int = 0, start_n: int = None) -> Optional[RunResult]:
    tf      = csv_info['tf']
    target  = SWEET_SPOTS.get(tf)
    if target is None:
        logger.warning("No sweet spot defined for '%s' — skipping.", tf)
        return None

    # Select N profile by phase order — first CSV gets full discovery range,
    # second CSV gets a tighter range since it seeds from phase 0 best params.
    profile_index = min(phase_index, len(PHASE_N_PROFILES) - 1)
    n_profile = PHASE_N_PROFILES[profile_index]

    out_dir = os.path.join('output', tf)
    os.makedirs(out_dir, exist_ok=True)

    state = PhaseState(tf=tf, csv_path=csv_info['path'], out_dir=out_dir,
                       n_profile=n_profile)
    n = start_n if start_n is not None else n_profile['start_n']

    print(f"\n{'='*62}")
    print(f"  PHASE {phase_index + 1}: {target['label']}")
    print(f"  CSV:   {csv_info['fname']}  ({csv_info['days']:.0f} trading days)")
    print(f"  Goal:  WR {target['wr_min']}–{target['wr_max']}%  |  "
          f"Signals/day {target['spd_min']:.0f}–{target['spd_max']:.0f}")
    print(f"  N:     start={n}  ceiling={n_profile['max_n']}  step={n_profile['n_step']}")
    print(f"  Seed:  {seed_params or 'fresh start'}")
    print(f"{'='*62}")

    while state.attempt < MAX_ATTEMPTS:
        state.attempt += 1

        ok = run_main(csv_info['path'], n, target['horizon'], out_dir,
                      params_path=seed_params)
        if not ok:
            logger.warning("Run failed — skipping attempt %d.", state.attempt)
            n = min(n + n_profile['n_step'], n_profile['max_n'])
            continue

        report = read_report(out_dir)
        if report is None:
            logger.warning("No report found after attempt %d.", state.attempt)
            continue

        result = evaluate(report, tf, n)
        state.history.append(result)
        print_result(state, result)

        if state.best is None or result.oos_score > state.best.oos_score:
            state.best = result
            logger.info("New phase best: WR=%.2f%% SPD=%.1f Score=%.4f",
                        result.wr, result.spd, result.oos_score)

        # Once we hit the sweet spot, count convergence attempts
        if result.in_sweet_spot:
            state.converge_count += 1
            if state.converge_count >= CONVERGENCE_HITS:
                logger.info("Converged! %d/%d sweet spot hits. Finalising.",
                            state.converge_count, CONVERGENCE_HITS)
                break

        # Determine next N
        n_new = choose_next_n(state, result, advisor)

        # Guard: if N isn't changing and we're stuck, force a step within ceiling
        if n_new == n and not result.in_sweet_spot:
            n_new = min(n + n_profile['n_step'], n_profile['max_n'])

        n = n_new
        # After first run always seed from the phase's own best
        seed_params = os.path.join(out_dir, 'best_report.json')

    if state.best:
        status = 'CONVERGED' if state.converge_count >= CONVERGENCE_HITS else 'BEST FOUND'
        logger.info("Phase %s complete [%s]: WR=%.2f%% SPD=%.1f N=%d Score=%.4f",
                    tf.upper(), status, state.best.wr, state.best.spd,
                    state.best.n, state.best.oos_score)

    return state.best


# ─── Entry Point ─────────────────────────────────────────────────────────────

def main():
    global MAX_ATTEMPTS
    import argparse
    p = argparse.ArgumentParser(description='Claude Gold — Meta-AutoResearch Pipeline')
    p.add_argument('--start-n', default=None, type=int,
                   help='Override starting N (default: phase0=10, phase1=5)')
    p.add_argument('--max-attempts', default=MAX_ATTEMPTS, type=int,
                   help='Max attempts per phase')
    args = p.parse_args()

    MAX_ATTEMPTS = args.max_attempts

    print("\n" + "="*62)
    print("  Claude Gold — Meta-AutoResearch Pipeline v1.0")
    print("  Scanning data/ for CSV files...")
    print("="*62)

    csvs = scan_csvs()
    if not csvs:
        print("\nERROR: No valid CSVs found in data/. Add your MT5 exports and retry.")
        sys.exit(1)

    print(f"\n  Found {len(csvs)} dataset(s) (running longest history first):")
    for c in csvs:
        ss = SWEET_SPOTS.get(c['tf'], {})
        label = ss.get('label', c['tf'].upper())
        print(f"    {c['tf'].upper()}: {c['fname']} — {c['days']:.0f} days → {label}")

    advisor   = OllamaAdvisor()
    phase_results = []
    seed_params   = None   # M1 phase seeds from M5 output

    for i, csv_info in enumerate(csvs):
        result = run_phase(
            csv_info    = csv_info,
            seed_params = seed_params,
            advisor     = advisor,
            phase_index = i,
            start_n     = args.start_n,
        )

        if result:
            phase_results.append((csv_info['tf'], result))

        # Pass this phase's best report as seed for the next phase
        phase_out = os.path.join('output', csv_info['tf'], 'best_report.json')
        if os.path.exists(phase_out):
            seed_params = phase_out

    # ── Final Summary ──────────────────────────────────────────────────────
    print("\n" + "="*62)
    print("  PIPELINE COMPLETE — Summary")
    print("="*62)

    for tf, r in phase_results:
        target = SWEET_SPOTS.get(tf, {})
        status = '✓ CONVERGED' if r.in_sweet_spot else '~ BEST FOUND'
        bench_wr  = target.get('benchmark_wr', '?')
        bench_spd = target.get('benchmark_spd', '?')
        print(f"""
  {tf.upper()} — {target.get('label', '')}  [{status}]
    Win Rate:    {r.wr:.2f}%      (benchmark: {bench_wr}%)
    Signals/Day: {r.spd:.1f}       (benchmark: {bench_spd})
    Stability:   {r.stability:.1f}%
    OOS Score:   {r.oos_score:.4f}
    Best N:      {r.n}
    Report:      output/{tf}/best_report.json""")

    print(f"\n  Next step: plug output/m5/best_report.json params into the EA's")
    print(f"  GetM5ConsensusBias() function and compare against benchmarks above.")
    print()


if __name__ == '__main__':
    main()
