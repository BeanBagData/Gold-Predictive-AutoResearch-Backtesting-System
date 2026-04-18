"""
core/ollama_advisor.py
======================
Connects to local Ollama instance for LLM-guided parameter mutations.
Injects explicit historical memory (Best/Worst) into the prompt to prevent duplicating bad searches.
"""
import json
import logging
import re
import requests

logger = logging.getLogger(__name__)

# Hard limits to prevent LLM indicator hacking
HARD_LIMITS = {
    'ema_fast': (3, 50),
    'ema_slow': (10, 200),
    'macd_fast': (2, 30),
    'macd_slow': (10, 60),
    'macd_signal': (2, 20),
    'adx_period': (5, 50),
    'adx_thresh': (10.0, 45.0),
    'stoch_k': (3, 40),
    'stoch_d': (2, 15),
    'stoch_smooth': (1, 15),
    'bb_period': (10, 60),
    'bb_stdev': (1.0, 4.0),
    'atr_period': (2, 40),
    'atr_mult': (0.0, 3.5),
    'rsi_period': (2, 30),
    'rsi_oversold': (10.0, 45.0),
    'rsi_overbought': (55.0, 90.0)
}

class OllamaAdvisor:
    def __init__(self, host: str = "http://localhost:11434", model: str = "Gemma4:e2b"):
        self.host = host
        self.model = model
        self.endpoint = f"{self.host}/api/generate"

    def _call_api(self, prompt: str) -> dict:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        try:
            response = requests.post(self.endpoint, json=payload, timeout=600)
            if response.status_code == 200:
                data = response.json()
                response_text = data.get("response", "")
                match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if match:
                    return json.loads(match.group(0))
            else:
                logger.error(f"Ollama API returned {response.status_code}: {response.text}")
        except Exception as e:
            logger.error(f"Ollama request failed: {e}")
        return {}

    def suggest_mutation(self, current_params: dict, metrics: dict, param_space: dict, iteration: int, stale_count: int = 0, target_bars: int = 1, top_best: list = None, top_worst: list = None) -> dict:
        keys = list(param_space.keys())
        ranges_str = json.dumps({k: [lo, hi] for k, (lo, hi, step) in param_space.items()})
        target_min = target_bars * 5

        # Format historical context for LLM
        best_str = "None"
        worst_str = "None"
        if top_best:
            best_str = json.dumps([{k: v for k, v in run.items() if k in param_space} for run in top_best], indent=2)
        if top_worst:
            worst_str = json.dumps([{k: v for k, v in run.items() if k in param_space} for run in top_worst], indent=2)

        stale_instruction = ""
        if stale_count > 10:
            stale_instruction = "FORCE DIVERGENCE: We are stuck. Move parameters to the opposite side of the range to find a new directional edge."

        prompt = f"""Output JSON with exactly these fields: {keys}

Ranges: {ranges_str}

Current best predictive TA params (Accuracy: {metrics.get('win_rate', 'N/A')}%):
{json.dumps(current_params)}

HISTORICAL CONTEXT:
Top 5 Best Parameter Sets (Benchmark - converge around these traits):
{best_str}

Top 5 Worst Parameter Sets (AVOID THESE AT ALL COSTS):
{worst_str}

CRITICAL RULES:
1. Output ONLY valid JSON, no text.
2. We are maximizing the probability of predicting the price direction {target_min} minutes ahead (candle close).
3. DO NOT output absurd mathematical values to "hack" the indicators.
4. ema_fast MUST BE LESS THAN ema_slow.
5. macd_fast MUST BE LESS THAN macd_slow.
6. DO NOT duplicate parameters found in the "Top 5 Worst" list.
{stale_instruction}

JSON:"""

        suggested = self._call_api(prompt)
        
        if suggested:
            for k, v in suggested.items():
                if k in param_space:
                    lo, hi, _ = param_space[k]
                    if isinstance(current_params.get(k), int):
                        suggested[k] = int(max(lo, min(hi, float(v))))
                    else:
                        suggested[k] = float(max(lo, min(hi, float(v))))
            
            if 'ema_fast' in suggested and 'ema_slow' in suggested:
                suggested['ema_fast'] = max(3, min(suggested['ema_fast'], suggested['ema_slow'] - 5))
                    
            if 'macd_fast' in suggested and 'macd_slow' in suggested:
                suggested['macd_fast'] = max(2, min(suggested['macd_fast'], suggested['macd_slow'] - 4))
                    
        return suggested

    def suggest_n_adjustment(self, current_n: int, metrics: dict, sweet_spot: dict,
                             history: list, verdict: str) -> dict:
        """Analyze run result and suggest next N for the pipeline meta-loop."""
        hist_summary = []
        for h in history[-6:]:
            hist_summary.append(f"  N={h['n']}: WR={h['wr']:.2f}% SPD={h['spd']:.1f} verdict={h['verdict']}")
        hist_str = '\n'.join(hist_summary) if hist_summary else '  (no history yet)'

        prompt = f"""You are optimizing an autoresearch pipeline for a gold scalping EA.
The pipeline runs parameter searches in cycles (N = number of Ollama advisor cycles).
Your job: suggest the best next N value to find the "sweet spot" parameters.

CURRENT RUN (N={current_n}):
  Win Rate:      {metrics.get('win_rate', 0):.2f}%
  Signals/Day:   {metrics.get('signals_per_day', 0):.1f}
  Stability:     {metrics.get('win_rate_stability', 0):.1f}%
  Verdict:       {verdict}

SWEET SPOT TARGETS:
  Win Rate:      {sweet_spot['wr_min']}% – {sweet_spot['wr_max']}%
  Signals/Day:   {sweet_spot['spd_min']} – {sweet_spot['spd_max']}
  Stability:     > Win Rate (must be higher than actual Win Rate)

RECENT HISTORY (last 6 runs):
{hist_str}

RULES:
- UNDERFITTING (low WR, low signals): increase N significantly (x1.5 to x2)
- OVERFITTING (high WR >target_max, very low signals): decrease N — too many cycles overfit
- TOO_RARE (WR in target but signals too low): moderately decrease N
- APPROACHING (close but not in sweet spot): increase N moderately (+5 to +15)
- SWEET_SPOT: nudge N by ±5 to see if score improves
- If a previous N produced better results, recommend returning to it
- N must be between 5 and {sweet_spot.get('max_n', 100)} — DO NOT exceed the max_n ceiling

Output ONLY valid JSON with exactly these fields:
{{"suggested_n": <integer>, "reasoning": "<one sentence>"}}

JSON:"""

        result = self._call_api(prompt)
        if result and 'suggested_n' in result:
            try:
                n = int(float(result['suggested_n']))
                result['suggested_n'] = max(5, min(200, n))
                return result
            except (ValueError, TypeError):
                pass
        return {}

    def adjust_search_space(self, current_space: dict, report_text: str) -> dict:
        prompt = f"""You are an AI Quant Algorithm adjusting a search space for a predictive trading model. 
Review the following Performance Report, which includes parameter correlations and the Top 5 Best/Worst combinations:
{report_text}

Your task is to update the search space boundaries [min, max, step] to force the next search generation into more profitable zones.

CRITICAL RULES FOR ADJUSTMENT:
1. Shift bounds lower if correlation is strongly Negative, or higher if strongly Positive.
2. DO NOT EXPAND TO INFINITY. You must provide realistic technical indicator constraints.
3. Look closely at the "TOP 5 BEST" and "TOP 5 WORST" runs. Shift boundaries away from values used by the worst runs, and towards values used by the best runs.
4. Make sure min is ALWAYS less than max.
5. DO NOT change the step sizes.
6. Output ONLY valid JSON where keys are parameters and values are [min, max, step].

Current Search Space:
{json.dumps(current_space, indent=2)}

Output the JSON with updated boundaries:"""

        suggested_space = self._call_api(prompt)
        validated_space = {}
        
        if suggested_space:
            for k, v in suggested_space.items():
                if k in current_space and isinstance(v, list) and len(v) == 3:
                    try:
                        lo, hi, step = float(v[0]), float(v[1]), float(v[2])
                        
                        if lo >= hi:
                            hi = lo + (step * 5)
                            
                        orig_lo, orig_hi, orig_step = current_space[k]
                        min_allowed_width = max(step * 4, (orig_hi - orig_lo) * 0.20)
                        absolute_floor = 0.0 if k == 'atr_mult' else 1.0
                        
                        if (hi - lo) < min_allowed_width:
                            mid = (hi + lo) / 2
                            lo = max(absolute_floor, mid - (min_allowed_width / 2))
                            hi = lo + min_allowed_width
                        else:
                            lo = max(absolute_floor, lo)
                            hi = max(lo + step, hi)
                            
                        if k in HARD_LIMITS:
                            min_allow, max_allow = HARD_LIMITS[k]
                            lo = max(min_allow, lo)
                            hi = min(max_allow, hi)
                            if lo >= hi:
                                lo = min_allow
                                hi = min_allow + (step * 2)

                        validated_space[k] = (lo, hi, step)
                    except (ValueError, TypeError):
                        pass
                        
        return validated_space