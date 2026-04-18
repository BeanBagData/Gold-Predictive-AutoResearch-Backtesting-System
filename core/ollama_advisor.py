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

    @staticmethod
    def _is_usable(obj: dict) -> bool:
        """
        Reject hallucinated nested garbage (e.g. {"downsampled": [{"key":...}]}).
        A usable response has at least one top-level value that is a scalar or a
        list whose first element is a number (adjust_search_space format).
        """
        if not isinstance(obj, dict) or not obj:
            return False
        for v in obj.values():
            if isinstance(v, (int, float, str, bool)):
                return True
            if isinstance(v, list) and v and isinstance(v[0], (int, float)):
                return True
        return False

    def _call_api(self, prompt: str) -> dict:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        try:
            response = requests.post(self.endpoint, json=payload, timeout=1200)
            if response.status_code != 200:
                logger.error("Ollama API returned %s: %s", response.status_code, response.text[:200])
                return {}

            response_text = response.json().get("response", "")

            # Guard: responses >8 KB are almost certainly hallucinated garbage.
            # Valid param JSON is <2 KB; log and bail early.
            if len(response_text) > 8000:
                logger.warning("Ollama response suspiciously large (%d chars) — skipping parse.",
                               len(response_text))
                return {}

            decoder = json.JSONDecoder()

            def _try_parse(text: str) -> dict:
                # Strategy A: whole text is valid JSON
                try:
                    obj = json.loads(text)
                    if self._is_usable(obj):
                        return obj
                except json.JSONDecodeError:
                    pass
                # Strategy B: raw_decode from first '{' — stops at end of first object
                start = text.find('{')
                if start >= 0:
                    try:
                        obj, _ = decoder.raw_decode(text, start)
                        if self._is_usable(obj):
                            return obj
                    except json.JSONDecodeError:
                        pass
                return {}

            # First attempt on raw response
            result = _try_parse(response_text)
            if result:
                return result

            # Strip markdown fences and retry
            cleaned = re.sub(r'```(?:json)?|```', '', response_text).strip()
            result = _try_parse(cleaned)
            if result:
                return result

            logger.warning("Ollama returned unusable JSON — skipping this cycle. "
                           "Snippet: %.120s", response_text[:120])

        except Exception as e:
            logger.error("Ollama request failed: %s", e)
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

        prompt = f"""You are a trading parameter optimiser. Output a FLAT JSON object — no nested arrays, no explanations, no markdown.

The JSON must have EXACTLY these keys with numeric values: {keys}

Allowed ranges per parameter: {ranges_str}

Current best params (Win Rate: {metrics.get('win_rate', 'N/A')}%):
{json.dumps(current_params)}

Top 5 Best (aim toward these):
{best_str}

Top 5 Worst (never output these values):
{worst_str}

Rules:
1. ema_fast < ema_slow
2. macd_fast < macd_slow
3. All values within the allowed ranges above
4. Do NOT copy any parameter set from the Worst list
{stale_instruction}

Output ONLY the flat JSON with the {len(keys)} numeric fields. No other text.
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

        delta_wr    = metrics.get('delta_wr', 0.0)
        delta_score = metrics.get('delta_oos_score', 0.0)
        stagnant    = abs(delta_wr) < 0.01 and abs(delta_score) < 0.01

        prompt = f"""You are optimizing an autoresearch pipeline for a gold scalping EA.
The pipeline runs parameter searches in cycles (N = number of Ollama advisor cycles).
Your job: suggest the best next N value to find the "sweet spot" parameters.

CURRENT RUN (N={current_n}):
  Win Rate:      {metrics.get('win_rate', 0):.2f}%  (change vs prev run: {delta_wr:+.3f}%)
  Signals/Day:   {metrics.get('signals_per_day', 0):.1f}
  Stability:     {metrics.get('win_rate_stability', 0):.1f}%
  OOS Score Δ:   {delta_score:+.4f}
  Verdict:       {verdict}
  STAGNATING:    {'YES — no improvement from last run' if stagnant else 'No'}

SWEET SPOT TARGETS:
  Win Rate:      {sweet_spot['wr_min']}% – {sweet_spot['wr_max']}%
  Signals/Day:   {sweet_spot['spd_min']} – {sweet_spot['spd_max']}
  Stability:     > Win Rate (must be higher than actual Win Rate)
  Streaks:       max_consec_correct must be >= 2 × max_consec_incorrect

RECENT HISTORY (last 6 runs):
{hist_str}

RULES:
- UNDERFITTING (low WR, low signals): increase N significantly (x1.5 to x2)
- OVERFITTING (high WR >target_max, very low signals): decrease N — too many cycles overfit
- TOO_RARE (WR in target but signals too low): moderately decrease N
- TOO_FREQUENT (signals WAY above spd_max — e.g. 87/day vs target 25-45): params are too loose/noisy.
  Increase N significantly (+20 to +40) to force exploration of tighter filter zones.
  Higher ADX threshold, higher bb_stdev, and lower stoch_smooth all reduce signal count.
  Do NOT use small step increments — this requires a large search jump.
- APPROACHING (close but not in sweet spot): increase N moderately (+5 to +15)
- QUALITY_FAIL (range correct but a quality gate failed): fine step ±1 to find the boundary
- SWEET_SPOT: nudge N by ±5 to see if score improves
- STAGNATING (change in WR and score both near zero): FORCE DIVERGENCE — jump N by at least +20,
  or return to a historically better N from the history list. Do not keep incrementing by 5.
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
1. POSITIVE CORRELATION + BEST NOT NEAR MAX: If a parameter shows positive correlation AND the current best
   value is NOT near the allowed maximum, RAISE the minimum boundary to force exploration of higher values.
   Do NOT just center tightly around the best value — push toward the max end of the range.
   Example: adx_thresh correlation=+0.33, best=34.87, max=45 → raise min to 37-40, keep max at 45.
2. NEGATIVE CORRELATION + BEST NOT NEAR MIN: Lower the maximum boundary to cut off high values.
   Example: macd_fast correlation=-0.25, best=19, min=8 → lower max to 22, keep min at 8.
3. AT MIN BOUND warning: If the best value equals or is very close to the current minimum, do NOT raise
   the minimum further. The parameter may need to explore LOWER. Instead, widen or lower the min.
4. DO NOT EXPAND TO INFINITY. Provide realistic technical indicator constraints.
5. Look at TOP 5 BEST and TOP 5 WORST. Shift away from worst param values, toward best param values.
6. Make sure min is ALWAYS less than max.
7. DO NOT change the step sizes.
8. MINIMUM RANGE WIDTH: New (max - min) must be at least 20% of the original range and at least 5 steps wide.
   Never trap a parameter in a tiny range — it kills exploration.
9. ema_slow: NEVER set min above 60. Locking ema_slow to a narrow high band prevents discovery of different
   trend-following speeds.
10. bb_stdev: Do not lock min at 1.0. If signals are too high, allow exploration of higher bb_stdev (wider
    bands = fewer signals). Keep min at 1.0 but allow max to rise toward 3.0-4.0.
11. Output ONLY valid JSON where keys are parameters and values are [min, max, step].

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
