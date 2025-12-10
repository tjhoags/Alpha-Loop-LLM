import difflib
from typing import Dict, List

import pandas as pd
from loguru import logger

from src.config.settings import get_settings

# You would likely integrate an LLM client here (e.g., OpenAI/Anthropic)
# from src.utils.llm_client import llm_query

class ResearchAutomation:
    def __init__(self):
        self.settings = get_settings()

    def competitive_moat_erosion_scanner(self, symbol: str, financials: pd.DataFrame) -> Dict:
        """Track market share data, pricing power, gross margin trends.
        Alert when moat metrics deteriorate 3+ quarters.
        """
        logger.info(f"Scanning moat erosion for {symbol}...")

        # 1. Gross Margin Trend
        if "gross_margin" not in financials.columns:
            return {"alert": False, "reason": "No margin data"}

        margins = financials["gross_margin"].sort_index()
        if len(margins) < 4:
            return {"alert": False, "reason": "Insufficient history"}

        # Check for 3 consecutive declines
        declining = True
        for i in range(1, 4):
            if margins.iloc[-i] >= margins.iloc[-(i+1)]:
                declining = False
                break

        if declining:
            return {
                "alert": True,
                "severity": "HIGH",
                "reason": "Gross Margin eroded for 3 consecutive quarters.",
            }

        return {"alert": False}

    def management_credibility_score(self, symbol: str, earnings_history: List[Dict]) -> float:
        """Track guidance vs. actual over time.
        Score management on a 0-100 credibility scale.
        """
        if not earnings_history:
            return 50.0 # Neutral start

        score = 50.0
        hits = 0
        misses = 0

        for q in earnings_history:
            guidance = q.get("guidance_eps")
            actual = q.get("actual_eps")

            if guidance is None or actual is None:
                continue

            # If they beat guidance, +points. If they miss, -points (heavier penalty).
            if actual >= guidance:
                score += 5
                hits += 1
            else:
                score -= 10 # Market hates misses
                misses += 1

        return max(0.0, min(100.0, score))

    def earnings_call_question_pattern_analyzer(self, transcript_text: str) -> Dict:
        """Identify tough questions and management dodges using LLM.
        """
        # Placeholder for LLM call
        # prompt = f"Analyze this transcript: {transcript_text}. List questions where management dodged."
        # response = llm_query(prompt)
        return {"dodges_detected": 0, "tough_questions": []}

    def diff_10k(self, current_10k_text: str, prior_10k_text: str) -> List[str]:
        """Automatically diff 10-K against prior year.
        Flag all meaningful changes in Risk Factors.
        """
        logger.info("Diffing 10-K filings...")
        # Use difflib for text comparison
        diff = difflib.unified_diff(
            prior_10k_text.splitlines(),
            current_10k_text.splitlines(),
            n=0,
        )

        # Filter for added lines (+) that look like risk factors
        new_risks = [line[1:] for line in diff if line.startswith("+") and len(line) > 50]
        return new_risks

    def variant_perception_identifier(self, estimates: List[Dict]) -> Dict:
        """Identify outliers in sell-side estimates.
        What does the most bullish/bearish analyst know?
        """
        if not estimates:
            return {}

        df = pd.DataFrame(estimates)
        mean_eps = df["eps_estimate"].mean()
        std_eps = df["eps_estimate"].std()

        outliers = df[
            (df["eps_estimate"] > mean_eps + 2*std_eps) |
            (df["eps_estimate"] < mean_eps - 2*std_eps)
        ]

        return {
            "consensus_mean": mean_eps,
            "variance": std_eps,
            "outliers": outliers.to_dict(orient="records"),
        }



