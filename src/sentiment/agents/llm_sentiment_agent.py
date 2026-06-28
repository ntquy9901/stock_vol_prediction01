"""
LLM Agent for Financial Sentiment Analysis
Using prompting instead of fine-tuning

Approach:
1. Few-shot learning with examples
2. Chain-of-thought reasoning
3. Rule-based post-processing
4. No fine-tuning required
"""

import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class SentimentLabel(Enum):
    """Sentiment labels"""
    POSITIVE = "Positive"
    NEGATIVE = "Negative"
    NEUTRAL = "Neutral"


@dataclass
class SentimentResult:
    """Sentiment analysis result"""
    sentiment_label: str
    sentiment_score: float
    reasoning: str
    confidence: float


class LLMSentimentAgent:
    """
    LLM Agent for financial sentiment analysis using prompting.

    Supports:
    - OpenAI GPT-4/GPT-3.5
    - Local LLM (Ollama, LM Studio)
    - Anthropic Claude
    - Google Gemini
    """

    def __init__(self, model_type: str = "openai", model_name: str = "gpt-4"):
        """
        Initialize LLM sentiment agent.

        Args:
            model_type: 'openai', 'anthropic', 'google', 'local'
            model_name: Specific model name (e.g., 'gpt-4', 'claude-3-sonnet')
        """
        self.model_type = model_type
        self.model_name = model_name
        self.few_shot_examples = self._load_few_shot_examples()

    def _load_few_shot_examples(self) -> List[Dict]:
        """
        Load few-shot examples for prompting.

        These examples teach the LLM financial sentiment patterns
        without requiring fine-tuning.
        """
        examples = [
            {
                "news": "Vietcombank reports record Q2 2026 profit of 9 trillion VND, up 20% YoY, exceeding analyst expectations",
                "sentiment": "Positive",
                "reasoning": "Key indicators: 'record profit', 'up 20% YoY', 'exceeding expectations' - all positive financial performance signals"
            },
            {
                "news": "Housing Development Bank misses Q2 profit targets, net profit down 5% YoY due to rising bad debts",
                "sentiment": "Negative",
                "reasoning": "Key indicators: 'misses targets', 'down 5% YoY', 'rising bad debts' - all negative financial performance signals"
            },
            {
                "news": "Techcombank maintains steady loan growth of 15% YoY in Q2 2026",
                "sentiment": "Neutral",
                "reasoning": "Key indicators: 'maintains steady', '15% growth' - stable performance without strong positive/negative signals"
            },
            {
                "news": "Vinhomes launches luxury real estate project in Hanoi targeting high-end buyers",
                "sentiment": "Positive",
                "reasoning": "Key indicators: 'launches', 'luxury', 'high-end' - business expansion and premium positioning"
            },
            {
                "news": "Masan Group experiences increased competition from foreign consumer goods companies entering Vietnamese market",
                "sentiment": "Negative",
                "reasoning": "Key indicators: 'increased competition', 'foreign companies entering' - market share risk and competitive pressure"
            },
            {
                "news": "Vinhomes prepares to close dividend record date with 60% cash payout (6,000 VND per share)",
                "sentiment": "Positive",
                "reasoning": "Key indicators: 'dividend', '60% cash payout', '6,000 VND per share' - shareholder return and cash distribution"
            }
        ]
        return examples

    def _create_prompt(self, news_text: str) -> str:
        """
        Create prompt with few-shot examples and chain-of-thought.

        Prompt structure:
        1. System instruction
        2. Few-shot examples
        3. Chain-of-thought template
        4. Current news to analyze
        """
        prompt = f"""You are a financial sentiment analysis expert specializing in Vietnamese stock market news.

Your task is to analyze financial news and determine if the sentiment is Positive, Negative, or Neutral.

Follow these rules:
1. Look for key financial indicators: earnings beats/misses, analyst upgrades/downgrades, partnerships, expansion, warnings
2. Consider the CONTEXT: Why is this news mentioned? What does it mean for investors?
3. Use chain-of-thought reasoning to explain your decision
4. Output in JSON format with sentiment, score, and reasoning

Few-Shot Examples (learn from these patterns):

"""

        # Add few-shot examples
        for i, example in enumerate(self.few_shot_examples, 1):
            prompt += f"""Example {i}:
News: {example['news']}
Sentiment: {example['sentiment']}
Reasoning: {example['reasoning']}

"""

        prompt += f"""Now analyze this news:

News: {news_text}

Provide your analysis in this JSON format:
{{
    "sentiment": "Positive/Negative/Neutral",
    "score": <number from -1.0 to +1.0>,
    "reasoning": "<step-by-step reasoning>",
    "confidence": <number from 0.0 to 1.0>
}}

Remember:
- Positive sentiment (+0.1 to +1.0): Good news, growth, expansion, upgrades, dividends, partnerships
- Negative sentiment (-0.1 to -1.0): Bad news, declines, misses, downgrades, warnings, competition
- Neutral sentiment (-0.1 to +0.1): Stable performance, routine operations, balanced info

Analyze now:"""

        return prompt

    def _create_rule_based_sentiment(self, news_text: str) -> Optional[str]:
        """
        Rule-based sentiment analysis as fallback/enhancement.

        This catches obvious patterns that LLM might miss.
        """
        news_lower = news_text.lower()

        # Strong positive patterns
        if any(word in news_lower for word in [
            "record profit", "record earnings", "exceptional growth",
            "upgrade", "buy recommendation", "outperform",
            "dividend", "payout", "shareholder return",
            "partnership", "strategic alliance", "acquisition",
            "expansion", "launches", "new product", "new store"
        ]):
            return "Positive"

        # Strong negative patterns
        if any(word in news_lower for word in [
            "misses earnings", "misses targets", "profit decline",
            "downgrade", "sell recommendation", "underperform",
            "warning", "concern", "risk", "disruption",
            "competition", "market share loss", "pressure",
            "bad debt", "npl", "default", "bankruptcy"
        ]):
            return "Negative"

        # Neutral patterns
        if any(word in news_lower for word in [
            "maintains", "steady", "stable", "continues",
            "routine", "normal operations", "balanced"
        ]):
            return "Neutral"

        return None

    def analyze_text(self, news_text: str, use_rules: bool = True) -> SentimentResult:
        """
        Analyze sentiment of news text using LLM + prompting.

        Args:
            news_text: Financial news text
            use_rules: Whether to use rule-based enhancement

        Returns:
            SentimentResult with label, score, reasoning, confidence
        """
        print(f"[LLM Agent] Analyzing news...")

        # Step 1: Try rule-based first (fast, accurate for obvious cases)
        if use_rules:
            rule_sentiment = self._create_rule_based_sentiment(news_text)
            if rule_sentiment:
                print(f"[Rule-Based] Sentiment: {rule_sentiment}")

                # Calculate score based on rule
                if rule_sentiment == "Positive":
                    score = 0.7
                    confidence = 0.9
                elif rule_sentiment == "Negative":
                    score = -0.7
                    confidence = 0.9
                else:  # Neutral
                    score = 0.0
                    confidence = 0.8

                return SentimentResult(
                    sentiment_label=rule_sentiment,
                    sentiment_score=score,
                    reasoning=f"Rule-based matching: {rule_sentiment} keywords detected",
                    confidence=confidence
                )

        # Step 2: If no rule match, use LLM with few-shot prompting
        prompt = self._create_prompt(news_text)
        llm_response = self._call_llm(prompt)

        # Step 3: Parse LLM response
        result = self._parse_llm_response(llm_response)

        print(f"[LLM] Sentiment: {result.sentiment_label} (Score: {result.sentiment_score:+.2f})")
        print(f"[LLM] Reasoning: {result.reasoning[:100]}...")

        return result

    def _call_llm(self, prompt: str) -> str:
        """
        Call LLM API (to be implemented based on model type).

        For now, returns mock response for demonstration.
        """
        # TODO: Implement actual API call
        # Options:
        # 1. OpenAI API (GPT-4/GPT-3.5)
        # 2. Anthropic API (Claude)
        # 3. Google API (Gemini)
        # 4. Local LLM (Ollama, LM Studio)

        if self.model_type == "openai":
            return self._call_openai(prompt)
        elif self.model_type == "anthropic":
            return self._call_anthropic(prompt)
        elif self.model_type == "local":
            return self._call_local_llm(prompt)
        else:
            # Mock response for demo
            return self._mock_llm_response(prompt)

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI GPT API."""
        try:
            import openai
            client = openai.OpenAI()

            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a financial sentiment analysis expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,  # Deterministic
                response_format={"type": "json_object"}
            )

            return response.choices[0].message.content

        except ImportError:
            print("[ERROR] openai package not installed. Install: pip install openai")
            return self._mock_llm_response(prompt)
        except Exception as e:
            print(f"[ERROR] OpenAI API call failed: {e}")
            return self._mock_llm_response(prompt)

    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic Claude API."""
        try:
            import anthropic

            client = anthropic.Anthropic()
            response = client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                temperature=0.0,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            return response.content[0].text

        except ImportError:
            print("[ERROR] anthropic package not installed. Install: pip install anthropic")
            return self._mock_llm_response(prompt)
        except Exception as e:
            print(f"[ERROR] Anthropic API call failed: {e}")
            return self._mock_llm_response(prompt)

    def _call_local_llm(self, prompt: str) -> str:
        """Call local LLM (Ollama or LM Studio)."""
        try:
            import requests

            # Ollama endpoint
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0}
                },
                timeout=60
            )

            return response.json()["response"]

        except Exception as e:
            print(f"[ERROR] Local LLM call failed: {e}")
            print("[INFO] Make sure Ollama is running: ollama serve")
            return self._mock_llm_response(prompt)

    def _mock_llm_response(self, prompt: str) -> str:
        """
        Mock LLM response for demonstration.

        This simulates what a real LLM would return.
        """
        # Extract news from prompt
        news_start = prompt.find("News: ") + 6
        news_end = prompt.find("\n\nProvide your analysis")
        news_text = prompt[news_start:news_end].strip()

        # Simple keyword matching for mock
        news_lower = news_text.lower()

        # Check for positive indicators
        positive_keywords = ["record", "exceeding", "launches", "partnership", "dividend", "expansion"]
        negative_keywords = ["misses", "down", "decline", "warning", "competition", "pressure", "disruption"]
        neutral_keywords = ["maintains", "steady", "stable", "continues", "balanced"]

        positive_count = sum(1 for kw in positive_keywords if kw in news_lower)
        negative_count = sum(1 for kw in negative_keywords if kw in news_lower)
        neutral_count = sum(1 for kw in neutral_keywords if kw in news_lower)

        # Determine sentiment
        if positive_count > negative_count and positive_count > neutral_count:
            sentiment = "Positive"
            score = 0.7
            reasoning = "Detected positive keywords indicating growth or good performance"
        elif negative_count > positive_count and negative_count > neutral_count:
            sentiment = "Negative"
            score = -0.7
            reasoning = "Detected negative keywords indicating decline or risks"
        elif neutral_count > positive_count and neutral_count > negative_count:
            sentiment = "Neutral"
            score = 0.0
            reasoning = "Detected neutral keywords indicating stable performance"
        else:
            # Default to slightly negative if unclear
            sentiment = "Negative"
            score = -0.3
            reasoning = "Unclear sentiment, defaulting to negative (cautious approach)"

        response = {
            "sentiment": sentiment,
            "score": score,
            "reasoning": reasoning,
            "confidence": 0.8
        }

        return json.dumps(response, indent=2)

    def _parse_llm_response(self, response: str) -> SentimentResult:
        """
        Parse LLM response into SentimentResult.

        Handles JSON parsing and error cases.
        """
        try:
            # Try to parse as JSON
            data = json.loads(response)

            sentiment = data.get("sentiment", "Neutral")
            score = float(data.get("score", 0.0))
            reasoning = data.get("reasoning", "")
            confidence = float(data.get("confidence", 0.5))

            return SentimentResult(
                sentiment_label=sentiment,
                sentiment_score=score,
                reasoning=reasoning,
                confidence=confidence
            )

        except json.JSONDecodeError:
            # Fallback: parse from text response
            print("[WARN] Could not parse JSON, using text parsing")

            response_lower = response.lower()

            if "positive" in response_lower:
                sentiment = "Positive"
                score = 0.6
            elif "negative" in response_lower:
                sentiment = "Negative"
                score = -0.6
            else:
                sentiment = "Neutral"
                score = 0.0

            return SentimentResult(
                sentiment_label=sentiment,
                sentiment_score=score,
                reasoning=response[:200],
                confidence=0.5
            )


def test_llm_agent():
    """Test LLM agent with sample news."""
    print("=" * 70)
    print("TESTING LLM SENTIMENT AGENT")
    print("=" * 70)

    # Initialize agent
    agent = LLMSentimentAgent(model_type="mock")

    # Test cases
    test_news = [
        "Vietcombank reports record Q2 2026 profit of 9 trillion VND, up 20% YoY, exceeding analyst expectations",
        "Housing Development Bank misses Q2 profit targets, net profit down 5% YoY due to rising bad debts",
        "Techcombank maintains steady loan growth of 15% YoY in Q2 2026"
    ]

    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    for i, news in enumerate(test_news, 1):
        print(f"\n[Test {i}]")
        print(f"News: {news[:70]}...")
        result = agent.analyze_text(news, use_rules=True)
        print(f"Result: {result.sentiment_label} (Score: {result.sentiment_score:+.2f})")
        print(f"Reasoning: {result.reasoning}")
        print("-" * 70)


if __name__ == "__main__":
    test_llm_agent()
