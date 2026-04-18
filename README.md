# Gold — Predictive AutoResearch & Backtesting System

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://makeapullrequest.com)

An AI-driven parameter optimisation pipeline for building directional bias signals from OHLCV price data. Uses a local Ollama LLM to guide a genetic-style search toward the statistically optimal indicator combination for any asset and timeframe.

Originally built for XAUUSD (Gold) scalping on MT4/MT5/Tradeview etc, but the backtester and pipeline are fully generic — drop in any MT5 CSV export and it works. 

Inspired by Karpathy's Autoresearch https://github.com/karpathy/autoresearch

For easy adaption, clone the repo, use a CLI or IDE like Claude Desktop App with Code, link the repo folder and intruct your LLM to make the logical changes to suit your asset.
You will want to provide an example of your own CSV data to ensure the correct parsing of the data. 

---

## Quick Start

> Complete setup takes about 10 minutes. Ollama must be running before you start.

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/claude-gold.git
cd claude-gold

# 2. Install Python dependencies
pip install pandas numpy requests

# 3. Install Ollama (see Ollama Setup section below)
#    Then pull the model:
ollama pull gemma4:e2b

# 4. Place your MT5 CSV export(s) in the data/ folder
#    Example: data/XAUUSD_M5_20241108_20260410.csv

# 5. Run a quick sanity check
python main.py --mode debug --horizon 3

# 6. Run a full automated pipeline (both CSVs at once)
python pipeline.py

# 7. Run a single focused search manually
python main.py --mode auto --horizon 3 --n 100
```

Results are saved to `output/<timeframe>/best_report.json` after every improvement.

---

## Table of Contents

1. [What This System Does](#what-this-system-does)
2. [Prerequisites](#prerequisites)
3. [Ollama Setup](#ollama-setup)
4. [File Structure](#file-structure)
5. [How It Works — Technical Overview](#how-it-works--technical-overview)
6. [Preparing Your Data](#preparing-your-data)
7. [Running a Single Search — main.py](#running-a-single-search--mainpy)
8. [Running the Full Pipeline — pipeline.py](#running-the-full-pipeline--pipelinepy)
9. [Finding the Sweet Spot](#finding-the-sweet-spot)
10. [Gold (XAUUSD) Parameters — Reference](#gold-xauusd-parameters--reference)
11. [Adapting for Other Assets](#adapting-for-other-assets)
12. [Configuration Reference](#configuration-reference)
13. [Troubleshooting](#troubleshooting)
14. [Contributing](#contributing)
15. [License](#license)

---

## What This System Does

Most trading system optimisers brute-force every combination of indicator parameters and pick the highest backtest return. This creates curve-fitted systems that fail in live trading.

This system takes a different approach:

**It asks a different question.** Instead of "which params made the most money historically?", it asks: *"Which indicator combination most reliably predicts whether the close of the next 15-minute candle will be higher or lower than the current close?"*

A system that can answer that question correctly 53–55% of the time, consistently, across both in-sample and out-of-sample data, across both bullish and bearish moves, with a stable win rate over time — is a system that has found a genuine statistical edge. The EA then trades around that edge.

**What the backtester measures:**
- Win rate (directional prediction accuracy)
- Signal frequency (how often the edge fires)
- Directional balance (works both long and short)
- Temporal stability (edge holds across time, not just lucky dates)
- Streak quality (consecutive correct > consecutive incorrect)

**What the optimiser does:**
- Hierarchically searches EMA, MACD, ADX, Stochastic, Bollinger Bands, and ATR combinations
- Uses a local Ollama LLM to suggest intelligent mutations rather than random guesses
- Splits data 80/20 into in-sample and out-of-sample to detect and reject overfitting
- Automatically adjusts the number of search cycles to converge on the sweet spot

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | 3.12 recommended |
| pandas | Any recent | `pip install pandas` |
| numpy | Any recent | `pip install numpy` |
| requests | Any recent | `pip install requests` |
| Ollama | Latest | See setup below |
| MT5 CSV data | — | Exported from MetaTrader 5 |

No GPU required. Ollama runs on CPU. A modern 8-core machine will handle everything comfortably.

---

## Ollama Setup

Ollama runs a local LLM server that the system uses to intelligently guide parameter mutations. It is not required — if Ollama is unavailable the system falls back to algorithmic logic — but results are significantly better with it running.

### 1. Install Ollama

**Windows:**
Download and run the installer from [ollama.com/download](https://ollama.com/download).

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Start the Ollama server

```bash
ollama serve
```

This starts the local API on `http://localhost:11434`. Keep this running in a separate terminal while using this system.

### 3. Pull the model

The system is configured to use `gemma4:e2b` by default (fast, low memory, excellent reasoning for structured JSON tasks):

```bash
ollama pull gemma4:e2b
```

**Alternative models** (change `model` in `core/ollama_advisor.py`):

| Model | Command | Speed | Quality | RAM |
|---|---|---|---|---|
| `gemma4:e2b` | `ollama pull gemma4:e2b` | Fast | Excellent | 7.2 GB |
| `gemma3:12b` | `ollama pull gemma3:12b` | Moderate | Better | 8 GB |
| `llama3.2:3b` | `ollama pull llama3.2:3b` | Very fast | Decent | 3 GB |
| `mistral:7b` | `ollama pull mistral:7b` | Moderate | Good | 6 GB |
| `qwen2.5:7b` | `ollama pull qwen2.5:7b` | Moderate | Good | 5 GB |

To change the model, edit line 36 of `core/ollama_advisor.py`:
```python
def __init__(self, host: str = "http://localhost:11434", model: str = "gemma4:e2b"):
```

### 4. Verify Ollama is working

```bash
ollama list
```

You should see your pulled model listed. The system will log a warning and fall back to algorithmic suggestions if the API is unreachable.

---

## File Structure

```
DIR/
│
├── pipeline.py                  ← Meta-pipeline: runs both CSVs sequentially,
│                                   auto-adjusts N, finds sweet spot
│
├── main.py                      ← Single-run entry point: debug / manual / auto modes
│
├── README.md                    ← This file
│
├── data/
│   ├── XAUUSD_M5_...csv         ← MT5 export: M5 data (longer history, runs first)
│   ├── XAUUSD_M1_...csv         ← MT5 export: M1 data (shorter history, runs second)
│   ├── loader.py                ← Parses MT5 CSV format (tab or comma separated)
│   └── feature_engineering.py  ← Computes all TA indicators from OHLCV
│
├── core/
│   ├── gold_autoresearch.py     ← v2.6 Discovery Mode genetic optimiser
│   └── ollama_advisor.py        ← Ollama LLM interface for parameter mutations
│
├── backtest/
│   ├── gold_backtester.py       ← Vectorised signal-vs-outcome evaluator
│   └── performance.py           ← Report generation and meta-analysis text
│
└── output/
    ├── m5/
    │   └── best_report.json     ← Best params found on M5 data
    └── m1/
        └── best_report.json     ← Best params found on M1 data (seeded from M5)
```

### Key file roles

**`pipeline.py`** — The master controller. Detects all CSVs in `data/`, identifies their timeframes from the filename or bar intervals, sorts them by history length (longest first), then runs each through an N-adjustment loop that converges on the sweet spot. Use this for a fully automated end-to-end run.

**`main.py`** — The individual run runner. Use this when you want to run a specific N value manually, debug your data loading, or resume from a known starting point.

**`core/gold_autoresearch.py`** — The optimiser brain. Runs a hierarchical phased search (Phase 1: trend params → Phase 2: momentum params → Phase 3: volatility params), calling Ollama every 15 generations for LLM-guided mutations and every 3 Ollama cycles for search space adjustment.

**`data/feature_engineering.py`** — Computes EMA, RSI, MACD, Stochastic, ATR, ADX, and Bollinger Bands from raw OHLCV. The prediction target is `close[t + horizon_bars] > close[t]` — a clean binary directional question anchored to the current bar's close.

---

## How It Works — Technical Overview

### The Prediction Problem

For each bar `t` in the dataset, the system generates a directional signal (+1 Buy, -1 Sell, 0 No Signal) based on the indicator state at `close[t]`. It then checks whether `close[t + N]` is above or below `close[t]`.

```
Bar t:   indicators fire → signal = +1 (predict UP)
Bar t+3: close[t+3] > close[t] → prediction was CORRECT ✓
```

For M5 data with `--horizon 3`, this is a 15-minute directional prediction.
For M1 data with `--horizon 15`, this is also a 15-minute directional prediction.

Both measure the same real-world question from different granularities.

### The Scoring Function

A raw win rate alone is not enough. A system that signals once a week with 60% accuracy is not tradeable. The composite score rewards:

```
score = accuracy × frequency × directional_balance × stability × volume_factor
```

| Component | What it measures | Why it matters |
|---|---|---|
| `accuracy` | `(win_rate - 50%) × 100` | Edge above coin flip |
| `frequency` | `log10(signals_per_day)` | Must be frequent enough to trade |
| `directional_balance` | Penalises lopsided up/down ratio | Prevents one-sided bias |
| `stability` | Rolling win rate std deviation | Edge must be consistent over time |
| `volume_factor` | Penalty if signals < daily floor | Forces tradeable frequency |

### The Search Strategy

The optimiser uses a hierarchical phased search:

- **Phase 1** (generations 0–30): Searches EMA trend, ADX period/threshold, MACD fast/slow
- **Phase 2** (generations 31–60): Adds Stochastic k/d/smooth, MACD signal
- **Phase 3** (generations 61+): Adds Bollinger Bands period/stdev, ATR period/multiplier

Each generation mutates the current best parameters using Gaussian noise (30% mutation rate per parameter). The best result across both in-sample and out-of-sample sets is tracked separately to detect and reject overfitting.

Every 15 generations, Ollama analyses the top 5 best and worst parameter sets and suggests an intelligent mutation. Every 45 generations, Ollama narrows the search space bounds toward the productive regions.

### IS/OOS Split

Data is split 80% in-sample / 20% out-of-sample. A result is only accepted as "best" if it improves the **out-of-sample** score — ensuring the discovered edge generalises beyond the training data.

---

## Preparing Your Data

### Exporting from MetaTrader 5

1. Open the chart for your instrument and timeframe
2. Go to **File → Save As** or use the **History Centre**
3. Export as CSV with columns: `DATE, TIME, OPEN, HIGH, LOW, CLOSE, TICKVOL`
4. Place the file in the `data/` folder

The loader handles both tab-separated and comma-separated formats automatically.

### Filename convention

MT5 exports include the timeframe in the filename (e.g., `XAUUSD_M5_20241108...csv`). The pipeline reads `_M1_`, `_M5_`, `_M15_` etc. from the filename to identify the timeframe without reading the full file. If your filename doesn't follow this convention, the system falls back to reading the actual bar intervals.

### How much data do you need?

| Timeframe | Minimum | Recommended | Notes |
|---|---|---|---|
| M1 | 1 month | 3–6 months | More than 6 months becomes slow to load |
| M5 | 6 months | 12–18 months | More data = more regime diversity = more robust params |
| M15 | 12 months | 24 months | |
| H1 | 2 years | 4 years | |

More data is better only up to a point. If the instrument has fundamentally changed behaviour (e.g., a regime change due to a central bank policy shift), older data may hurt more than it helps.

---

## Running a Single Search — main.py

### Modes

**Debug** — verify your data loads correctly and see a baseline result with default params:
```bash
python main.py --mode debug --horizon 3
```

**Auto** — run a full parameter search:
```bash
python main.py --mode auto --horizon 3 --n 100
```

**Auto with resume** — continue from the last best result:
```bash
python main.py --mode auto --horizon 3 --n 50 --resume
```

**Manual** — run the backtester with specific params from a saved report:
```bash
python main.py --mode manual --params output/m5/best_report.json --horizon 3
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--mode` | `debug` | `debug`, `auto`, or `manual` |
| `--horizon` | `3` | Bars ahead to predict (M5: use 3 for 15min, M1: use 15 for 15min) |
| `--n` | `30` | Number of Ollama advisor cycles (15 experiments each = N×15 total runs) |
| `--resume` | off | Load best params from `output/best_report.json` as starting point |
| `--params` | — | Path to a specific `best_report.json` to seed from |
| `--out` | `output` | Directory to save reports |
| `--seed` | `42` | Random seed for reproducibility |

### Understanding N

`--n` controls the number of Ollama advisor cycles. Each cycle runs 15 parameter experiments, so `--n 100` = 1500 total experiments. Between cycles, Ollama analyses the results and suggests the next mutation. Every 3 cycles, Ollama also adjusts the search space bounds.

For M5 data, the sweet spot is typically found between N=70–100 on a fresh discovery run.
For M1 data, it is typically found in the same range but the pipeline will seed from M5 so it converges faster.

---

## Running the Full Pipeline — pipeline.py

The pipeline automates everything: CSV detection, timeframe identification, sequential phase execution, N-adjustment, sweet spot convergence, and cross-phase seeding.

### Basic usage

Place both CSV files in `data/` and run:
```bash
python pipeline.py
```

### What happens

```
1. Scans data/ — finds M5 (17 months) and M1 (3.5 months) CSVs
2. Sorts by history length: M5 first, M1 second

PHASE 1 — M5 (fresh start)
   N=10 → evaluate → UNDERFITTING → N=20
   N=20 → evaluate → APPROACHING  → N=21 (step collapses to 1)
   N=21 → evaluate → SWEET_SPOT   → converge count: 1/5
   N=22 → evaluate → SWEET_SPOT   → converge count: 2/5
   ... (5 hits) → Phase 1 complete → save output/m5/best_report.json

PHASE 2 — M1 (seeded from M5 best params)
   N=5  → evaluate → APPROACHING  → N=6 (already close due to seeding)
   N=6  → evaluate → SWEET_SPOT   → converge count: 1/5
   ... (5 hits) → Phase 2 complete → save output/m1/best_report.json

3. Final summary printed with comparison to benchmarks
```

### N ceiling by phase

| Phase | CSV | Start N | Ceiling | Step (far) | Step (near) |
|---|---|---|---|---|---|
| Phase 1 | Longest history | 10 | **150** | 10 | **1** |
| Phase 2 | Shorter history | 5 | **25** | 5 | **1** |

Step automatically collapses to 1 when the result is within 1.5% WR of the target range and 75%+ of the signal floor. This allows Ollama to make fine single-cycle comparisons and pick the globally best N within the sweet spot.

### Pipeline arguments

```bash
python pipeline.py --start-n 20        # override starting N for both phases
python pipeline.py --max-attempts 30   # increase total attempts per phase
```

---

## Finding the Sweet Spot

The sweet spot is a combination of win rate, signal frequency, and statistical quality that produces a genuinely tradeable edge. It is not simply "highest win rate" — a 70% win rate from 3 signals per day is useless live.

### The five quality gates

A result must pass **all five** to count as a sweet spot hit:

| Gate | Condition | What failure means |
|---|---|---|
| Win rate range | `wr_min ≤ WR ≤ wr_max` | Edge is too weak or overfitted |
| Signal frequency | `spd_min ≤ SPD ≤ spd_max` | Too rare to trade or too noisy |
| Stability vs WR | `stability% > win_rate%` | Edge relies on lucky streaks, inconsistent over time |
| Directional balance | `up_wr > 51% AND down_wr > 51%` | One-sided bias hiding overall WR |
| Streak quality | `consec_correct > consec_incorrect` | Losing runs outnumber winning runs |

### Recognising overfitting

| Symptom | What it looks like | Action |
|---|---|---|
| Ghost hunter | WR > 60%, signals < 5/day | Reduce N — too many cycles overfit to rare events |
| Lucky date bias | IS win rate >> OOS win rate | Widen search space, more data |
| One-sided bias | up_wr = 61%, down_wr = 43% | System only works in one direction |
| Stability collapse | WR 54% but stability 30% | Edge is clustered in a few lucky windows |

### Recognising underfitting

| Symptom | What it looks like | Action |
|---|---|---|
| Random noise | WR ≈ 50%, any signal count | Increase N — not enough exploration |
| Volume problem | WR looks good but score low | Signal count below daily floor |
| Flat score | Score < 0.5 after 50+ generations | Try widening search space |

### Verdicts explained

| Verdict | Meaning | Pipeline response |
|---|---|---|
| `UNDERFITTING` | WR and/or SPD far below target | Large N jump (+2× step) |
| `APPROACHING` | Getting close (within 1.5% WR) | Step collapses to 1 |
| `TOO_RARE` | WR good but too few signals | Reduce N to last good N |
| `OVERFITTING` | High WR but ghost hunter | Binary search between good N and current |
| `OVER_OPTIMIZED` | WR above target range | Reduce N |
| `QUALITY_FAIL` | Range correct, quality gates fail | Step=1, Ollama selects best |
| `SWEET_SPOT` | All five gates pass | Count toward 5-hit convergence |

---

## Gold (XAUUSD) Parameters — Reference

These were discovered through extensive research on XAUUSD using an MT4/MT5 grid-hedge EA. They serve as the target benchmarks the pipeline is trying to beat.

### M5 Sweet Spot — "Institutional Flow"

The M5 edge captures steady directional momentum on the 6-hour trend cycle.

| Parameter | Value | Role |
|---|---|---|
| EMA Fast | 10 | Short-term trend direction |
| EMA Slow | 80 | 6-hour macro trend (80 × 5min = 400min) |
| MACD Fast | 22 | ~110 min momentum |
| MACD Slow | 60 | ~300 min momentum tide |
| MACD Signal | 6 | Signal smoothing |
| ADX Period | 6 | Short-period trend strength |
| ADX Threshold | 35.0 | Only trade when trend has ignition |
| Stochastic K | 19 | Timing oscillator period |
| Stochastic D | 2 | Signal line |
| Stochastic Smooth | 15 | Strong smoothing — filters whipsaws |
| RSI Period | 5 | Entry hook (hard-locked in EA) |
| RSI Oversold | 29.0 | Buy hook trigger |
| RSI Overbought | 79.0 | Sell hook trigger |

**Performance benchmarks:**

| Metric | Value |
|---|---|
| OOS Win Rate | 54.38% |
| IS Win Rate | 50.84% |
| IS/OOS Drop | 3.54% — very small, confirms robustness |
| Signals/Day | 42.1 (≈ 2 trades per hour) |
| Strategy type | Trend scalper |

### M1 Sweet Spot — "Momentum Ignition"

The M1 edge captures explosive velocity bursts — price moving more than its average distance per minute over a short lookback.

| Parameter | Value | Role |
|---|---|---|
| EMA Fast | 5 | Ultra-responsive trend |
| EMA Slow | 10 | Short-window trend direction |
| MACD Fast | 24 | ~24 min momentum |
| MACD Slow | 60 | ~60 min momentum tide |
| MACD Signal | 6 | Signal smoothing |
| ADX Period | 27 | Longer-period stability filter |
| ADX Threshold | 45.0 | High bar — only trades strong bursts |

**Performance benchmarks:**

| Metric | Value |
|---|---|
| OOS Win Rate | ~52.0% |
| Signals/Day | 116 (≈ 1 trade per 12 minutes) |
| Stability | 67% — very high for M1 |
| Strategy type | Momentum sniper |

### The "Diminishing Returns" wall

**M5:** When the AI pushed win rate above 53.5%, signals fell below 12/day. The system became a "ghost hunter" — only firing on rare news events. Not tradeable live.

**M1:** When win rate was pushed above 52.5%, signals fell below 40/day. Same problem at a different scale.

**Rule:** Any strategy with fewer than 12 signals/day on M5 or 40 signals/day on M1 should be treated as overfitted regardless of how good the win rate looks.

---

## Adapting for Other Assets

The backtester is completely generic. The only gold-specific elements are the sweet spot targets in `pipeline.py` and the RSI hard-lock in `gold_autoresearch.py`. Everything else works with any OHLCV data.

### Step 1 — Add your asset's sweet spot to pipeline.py

Open `pipeline.py` and add a new entry to `SWEET_SPOTS`:

```python
SWEET_SPOTS = {
    'm5': { ... },   # existing gold M5
    'm1': { ... },   # existing gold M1

    # Example: EUR/USD M15 trend following
    'm15': {
        'label':         'EURUSD M15 Trend Follow',
        'horizon':       4,        # 4 × 15min = 1 hour prediction
        'wr_min':        52.0,
        'wr_max':        54.5,
        'spd_min':       8.0,      # fewer signals expected on M15
        'spd_max':       20.0,
        'benchmark_wr':  0.0,      # unknown until you run it
        'benchmark_spd': 0.0,
    },
}
```

### Step 2 — Remove the RSI hard-lock (or adjust it)

The RSI 5/29/79 lock in `core/gold_autoresearch.py` is specific to the gold EA's entry logic. For a different asset or strategy, either remove it or change the values:

```python
# In core/gold_autoresearch.py
STATIC_EA_PARAMS = {
    # Remove or change these for non-gold assets:
    'rsi_period':    5,
    'rsi_oversold':  29.0,
    'rsi_overbought': 79.0,
}
```

To make RSI fully researchable, move it to `INITIAL_PARAM_SPACE`:
```python
INITIAL_PARAM_SPACE = {
    ...existing params...
    'rsi_period':    (3, 21, 1),
    'rsi_oversold':  (20.0, 40.0, 1.0),
    'rsi_overbought': (60.0, 85.0, 1.0),
}
```

And remove it from `INT_PARAMS` or `STATIC_EA_PARAMS` accordingly.

### Step 3 — Adjust the signal floor

In `core/gold_autoresearch.py`, the signal floor targets scalping frequency. Adjust for your instrument's natural activity:

```python
# Current settings (scalping):
signals_per_day_target = 60.0 if bars_per_day > 300 else 30.0

# For swing trading (H1/H4):
signals_per_day_target = 3.0 if bars_per_day > 20 else 1.0

# For index futures intraday (M5):
signals_per_day_target = 15.0  # SPX/NQ are more regulated in movement
```

### Asset-specific guidance

#### Forex (EUR/USD, GBP/USD, etc.)

- **Session awareness matters more.** London open and NY open produce very different behaviour. Consider filtering your data to session hours before backtesting.
- **Spread is larger relative to ATR** on exotic pairs — keep `atr_mult` lower.
- **Suggested horizon:** M5 → 3 bars (15 min), M15 → 4 bars (1 hour)
- **Expected WR range:** 51–54% (forex is highly efficient)
- **Expected signals:** 15–30/day on M5

#### S&P 500 / NASDAQ (SPX, NQ, US500)

- **Indices trend strongly during US session, reverse at open/close.** Consider session-filtered data.
- **Volatility regime shifts** (VIX spikes) create false signals — consider adding a volatility filter to the EA side.
- **Suggested horizon:** M5 → 3 bars (15 min), M15 → 8 bars (2 hours)
- **Expected WR range:** 52–56% (indices trend more cleanly than forex)
- **Expected signals:** 10–25/day on M5
- Sweet spot ADX threshold likely higher: 40–50 (only trade confirmed moves)

#### Crude Oil (WTI, BRENT)

- Very similar to gold — volatile, momentum-driven, responds to macro events.
- Start with the gold M5 sweet spot params as a baseline and refine.
- **Expected WR range:** 53–56%
- **Expected signals:** 20–40/day on M5

#### Crypto (BTC, ETH)

- 24/7 market with no weekend gaps — your M5 data will have no missing sessions.
- Much higher volatility — `adx_thresh` may need to be higher (50+) to filter noise.
- **Suggested horizon:** M5 → 3 bars, or M1 → 15 bars for scalping
- **Expected WR range:** 51–53% (crypto is very noisy)
- **Warning:** Crypto data often contains anomalous wicks and exchange-specific artefacts. Clean your data before use.

### Changing the prediction horizon

The `--horizon` argument sets how many bars ahead to predict. The sweet spot for a given instrument often depends on its natural "impulse duration" — how long a directional move typically lasts before reversing.

| Asset | Timeframe | Horizon | Prediction window | Notes |
|---|---|---|---|---|
| XAUUSD | M1 | 15 | 15 minutes | Scalping |
| XAUUSD | M5 | 3 | 15 minutes | Scalping |
| EURUSD | M5 | 6 | 30 minutes | Intraday |
| EURUSD | M15 | 4 | 1 hour | Intraday |
| SPX | M5 | 12 | 1 hour | Intraday |
| SPX | M15 | 4 | 1 hour | Intraday |
| BTC | M5 | 3 | 15 minutes | Scalping |
| BTC | H1 | 4 | 4 hours | Swing |

---

## Configuration Reference

### core/gold_autoresearch.py

```python
# Hard-locked EA params — change these to match your EA's fixed logic
STATIC_EA_PARAMS = {
    'rsi_period':    5,
    'rsi_oversold':  29.0,
    'rsi_overbought': 79.0,
}

# Search space bounds: (min, max, step)
INITIAL_PARAM_SPACE = {
    'ema_fast':    (3,  50,  1),
    'ema_slow':    (20, 200, 1),
    'macd_fast':   (5,  25,  1),
    'macd_slow':   (25, 80,  1),
    'macd_signal': (2,  15,  1),
    'adx_period':  (5,  20,  1),
    'adx_thresh':  (15.0, 60.0, 1.0),
    'stoch_k':     (3,  20,  1),
    'stoch_d':     (2,  5,   1),
    'stoch_smooth':(1,  15,  1),
    'bb_period':   (10, 60,  1),
    'bb_stdev':    (1.0, 3.0, 0.1),
    'atr_period':  (2,  30,  1),
    'atr_mult':    (0.5, 3.0, 0.1),
}

# Phase boundaries (generation numbers)
PHASE_1_KEYS = ['ema_fast', 'ema_slow', 'adx_thresh', 'adx_period', 'macd_fast', 'macd_slow']
PHASE_2_KEYS = ['stoch_k', 'stoch_d', 'stoch_smooth', 'macd_signal']
PHASE_3_KEYS = ['bb_period', 'bb_stdev', 'atr_period', 'atr_mult']

# Experiments per Ollama cycle (15 = reasonable; lower = more Ollama guidance)
EXPERIMENTS_PER_OLLAMA_CYCLE = 15

# Signal floor targets (per day)
signals_per_day_target = 60.0 if bars_per_day > 300 else 30.0
```

### pipeline.py — Sweet spot targets

```python
SWEET_SPOTS = {
    'm5': {
        'wr_min': 53.5, 'wr_max': 55.0,    # win rate target range
        'spd_min': 25.0, 'spd_max': 45.0,  # signals per day range
        'horizon': 3,                       # bars ahead to predict
    },
    'm1': {
        'wr_min': 51.5, 'wr_max': 52.5,
        'spd_min': 80.0, 'spd_max': 120.0,
        'horizon': 15,
    },
}

PHASE_N_PROFILES = [
    {'start_n': 10, 'max_n': 150, 'n_step': 10},  # Phase 0: first CSV
    {'start_n':  5, 'max_n':  25, 'n_step':  5},  # Phase 1: second CSV
]

CONVERGENCE_HITS = 5   # must hit sweet spot this many times to finalise
MAX_ATTEMPTS     = 25  # hard cap on attempts per phase
```

### core/ollama_advisor.py — Model and server

```python
class OllamaAdvisor:
    def __init__(self, host: str = "http://localhost:11434", model: str = "gemma4:e2b"):
```

Change `model` to any Ollama-compatible model. Change `host` if running Ollama on a different machine.

---

## Troubleshooting

### "No CSV found in data/"
Ensure your CSV file is in the `data/` subfolder and the filename does not contain any of the exclude patterns: `log`, `result`, `state`, `params`, `autoresearch`, `claude_gold`, `best_report`.

### "Multiple CSVs in data/" (when running main.py directly)
`main.py` expects exactly one CSV when run directly. Either remove one, or use `pipeline.py` which handles multiple CSVs automatically.

### "Ollama request failed"
- Ensure `ollama serve` is running in another terminal
- Verify the model is pulled: `ollama list`
- The system will fall back to algorithmic N-adjustment — results will still work, just less guided

### Score stays near 0 after many generations
- Win rate is near 50% — the indicator combination is not finding a real edge
- Try widening the `adx_thresh` range downward (allow weaker trends)
- Try reducing the signal floor to let the system find something first, then tighten
- Check your data is clean and sorted correctly: `python main.py --mode debug`

### Win rate is high but score is low
- Signal count is below the daily floor — the system is penalising low volume
- Lower `signals_per_day_target` temporarily to let the optimiser find a base, then raise it
- This is the "ghost hunter" problem — high ADX threshold filtering too aggressively

### IS win rate >> OOS win rate (overfitting)
- Large gap means the system memorised the training data
- The pipeline guards against this by only accepting improvements on OOS data
- If it persists, reduce `EXPERIMENTS_PER_OLLAMA_CYCLE` to force more Ollama guidance
- Consider shortening the in-sample period (change the 0.8 split in `main.py`)

### Pipeline runs M1 before M5
The pipeline sorts by number of trading days (longest first). If your M1 file covers a longer calendar period than your M5 file, it will run first. Ensure your M5 file covers a longer date range for the intended order.

### Results are not reproducible
Set `--seed` to a fixed value:
```bash
python main.py --mode auto --horizon 3 --n 100 --seed 42
```

---

## Output Files

After every improvement, the system saves `best_report.json` to the output directory. This file contains:

```json
{
  "timestamp": "2026-04-18 03:45:12",
  "generation": 847,
  "best_oos_score": 4.2819,
  "oos_metrics": {
    "win_rate": 54.21,
    "total_signals": 11403,
    "signals_per_day": 41.8,
    "up_win_rate": 53.7,
    "down_win_rate": 54.8,
    "consec_correct": 9,
    "consec_incorrect": 7,
    "win_rate_stability": 71.3,
    "score": 4.2819
  },
  "params": {
    "ema_fast": 10, "ema_slow": 80,
    "macd_fast": 22, "macd_slow": 60, "macd_signal": 6,
    "adx_period": 6, "adx_thresh": 35.0,
    "stoch_k": 19, "stoch_d": 2, "stoch_smooth": 15,
    "bb_period": 28, "bb_stdev": 1.8,
    "atr_period": 14, "atr_mult": 1.2,
    "rsi_period": 5, "rsi_oversold": 29.0, "rsi_overbought": 79.0
  },
  "history": [ ... all previous runs ... ]
}
```

These params map directly to indicator settings in your EA. Plug them into your EA's consensus function and compare live results against the backtested win rate.

---

## Contributing

Contributions are welcome! If you have an idea for improving the search logic, adding new indicators, or making the pipeline more robust:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

**Disclaimer:** This project is provided for educational and research purposes only. Backtested results do not guarantee future performance. The authors and contributors are not responsible for any financial losses incurred from using this software. Always test on a demo account before live deployment.
```
