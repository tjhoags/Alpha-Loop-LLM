"""
OpenAI/Claude AI Analysis Client
================================
AI-powered market analysis using LLMs.
"""

import os
import logging
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class OpenAIAnalyzer:
    """
    AI-powered analysis using OpenAI GPT or Anthropic Claude.
    
    Use cases:
    - Narrative analysis from text
    - Sentiment classification
    - Thesis generation
    - Research summarization
    """
    
    def __init__(
        self,
        provider: str = "openai",
        api_key: str = None
    ):
        self.provider = provider
        
        if provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("openai package required")
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            self.client = openai.OpenAI(api_key=self.api_key)
            self.model = "gpt-4-turbo-preview"
        elif provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("anthropic package required")
            self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            self.client = anthropic.Anthropic(api_key=self.api_key)
            self.model = "claude-3-sonnet-20240229"
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _call_llm(self, prompt: str, system_prompt: str = None) -> str:
        """Make LLM API call"""
        try:
            if self.provider == "openai":
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=2000
                )
                return response.choices[0].message.content
            
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    system=system_prompt or "",
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
        
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    def analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment of financial text.
        
        Returns:
            sentiment: "bullish", "bearish", "neutral"
            confidence: 0-100
            key_points: List of key points
        """
        system_prompt = """You are a financial sentiment analyst. Analyze the sentiment 
        of the provided text and return a JSON response with:
        - sentiment: "bullish", "bearish", or "neutral"
        - confidence: integer 0-100
        - key_points: list of 3-5 key points
        
        Return ONLY valid JSON, no other text."""
        
        response = self._call_llm(text, system_prompt)
        
        # Parse JSON response
        import json
        try:
            return json.loads(response)
        except:
            return {
                "sentiment": "neutral",
                "confidence": 50,
                "key_points": ["Unable to parse response"]
            }
    
    def extract_tickers(self, text: str) -> List[str]:
        """Extract ticker symbols from text"""
        system_prompt = """Extract all stock ticker symbols mentioned in the text.
        Return ONLY a JSON array of uppercase ticker symbols, nothing else.
        Example: ["NVDA", "AAPL", "CCJ"]"""
        
        response = self._call_llm(text, system_prompt)
        
        import json
        try:
            return json.loads(response)
        except:
            return []
    
    def generate_thesis(
        self,
        topic: str,
        context: str = ""
    ) -> Dict:
        """
        Generate investment thesis from a topic.
        
        Returns structured thesis with beneficiaries.
        """
        system_prompt = """You are an investment analyst specializing in identifying 
        multi-order beneficiaries of investment themes. Given a topic, generate a 
        structured thesis including:
        
        1. Core thesis (2-3 sentences)
        2. Order 0 beneficiaries (obvious, direct plays)
        3. Order 1 beneficiaries (sell-side consensus)
        4. Order 2-3 beneficiaries (hidden opportunities)
        5. Key catalysts and timeline
        6. Risk factors
        
        Return as JSON with these fields."""
        
        prompt = f"Topic: {topic}\n\nContext: {context}" if context else f"Topic: {topic}"
        
        response = self._call_llm(prompt, system_prompt)
        
        import json
        try:
            return json.loads(response)
        except:
            return {"error": "Unable to parse thesis", "raw": response}
    
    def summarize_research(
        self,
        text: str,
        max_points: int = 5
    ) -> Dict:
        """Summarize research document into key points"""
        system_prompt = f"""Summarize this financial research into:
        1. Main thesis (1 sentence)
        2. Up to {max_points} key investment points
        3. Recommended action (buy/sell/hold)
        4. Key risks
        
        Return as JSON."""
        
        response = self._call_llm(text, system_prompt)
        
        import json
        try:
            return json.loads(response)
        except:
            return {"error": "Unable to parse summary"}
    
    def analyze_narrative_stage(
        self,
        narrative: str,
        recent_mentions: List[str]
    ) -> Dict:
        """
        Analyze where a narrative is in its lifecycle.
        
        Stages: emergence, early_adoption, acceleration, mainstream, saturation
        """
        system_prompt = """Analyze the lifecycle stage of this investment narrative.
        
        Stages:
        - emergence: Just starting, few mentions
        - early_adoption: Growing interest, niche communities
        - acceleration: Rapid growth, Reddit DD posts
        - mainstream: Widely discussed, mainstream coverage
        - saturation: Everyone knows, likely peaked
        
        Return JSON with:
        - stage: one of the above
        - confidence: 0-100
        - reasoning: brief explanation
        - time_to_next_stage: estimated days/weeks"""
        
        prompt = f"""Narrative: {narrative}
        
        Recent mentions/sources:
        {chr(10).join(recent_mentions[:10])}"""
        
        response = self._call_llm(prompt, system_prompt)
        
        import json
        try:
            return json.loads(response)
        except:
            return {"stage": "unknown", "confidence": 0}


class SlackNotifier:
    """Simple Slack notification client"""
    
    def __init__(self, token: str = None, channel: str = None):
        self.token = token or os.getenv("SLACK_BOT_TOKEN")
        self.channel = channel or os.getenv("SLACK_CHANNEL_ALERTS", "#trading-alerts")
        
        try:
            from slack_sdk import WebClient
            self.client = WebClient(token=self.token)
            self.available = True
        except ImportError:
            self.available = False
            logger.warning("slack_sdk not installed")
    
    def send_message(self, message: str, channel: str = None) -> bool:
        """Send message to Slack channel"""
        if not self.available:
            logger.warning("Slack not available")
            return False
        
        try:
            self.client.chat_postMessage(
                channel=channel or self.channel,
                text=message
            )
            return True
        except Exception as e:
            logger.error(f"Slack send failed: {e}")
            return False
    
    def send_alert(
        self,
        title: str,
        message: str,
        severity: str = "info"
    ) -> bool:
        """Send formatted alert"""
        emoji = {
            "info": "ℹ️",
            "warning": "⚠️",
            "critical": "🚨",
            "success": "✅"
        }.get(severity, "📢")
        
        formatted = f"{emoji} *{title}*\n{message}"
        return self.send_message(formatted)


if __name__ == "__main__":
    # Demo
    if OPENAI_AVAILABLE:
        analyzer = OpenAIAnalyzer(provider="openai")
        
        # Sentiment analysis
        text = "NVDA reported record earnings driven by AI demand. Data center revenue up 400% YoY."
        sentiment = analyzer.analyze_sentiment(text)
        print(f"Sentiment: {sentiment}")
    else:
        print("OpenAI client unavailable - openai package not installed")

