"""
FinBERT Sentiment Analysis Model

Financial sentiment analysis using FinBERT (ProsusAI/finbert)
- Fine-tuned BERT model for financial text sentiment classification
- Supports English text (Vietnamese requires translation)
- Returns sentiment scores and classification (Positive/Neutral/Negative)
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentResult:
    """Result object for sentiment analysis"""

    def __init__(self, sentiment_score: float, sentiment_label: str,
                 positive_score: float, negative_score: float, neutral_score: float):
        self.sentiment_score = sentiment_score
        self.sentiment_label = sentiment_label
        self.positive_score = positive_score
        self.negative_score = negative_score
        self.neutral_score = neutral_score

    def __repr__(self):
        return (f"SentimentResult(label={self.sentiment_label}, "
                f"score={self.sentiment_score:.3f}, "
                f"pos={self.positive_score:.3f}, "
                f"neg={self.negative_score:.3f}, "
                f"neu={self.neutral_score:.3f})")


class FinBERTSentiment:
    """
    Financial sentiment analysis using FinBERT model.

    This class provides sentiment analysis capabilities specifically designed
    for financial text using the FinBERT model from ProsusAI.

    Attributes:
        model_name (str): HuggingFace model identifier
        device (torch.device): CPU or CUDA device
        labels (dict): Mapping from label IDs to sentiment classes
    """

    def __init__(self, model_name: str = "ProsusAI/finbert", device: Optional[str] = None):
        """
        Initialize FinBERT sentiment analyzer.

        Args:
            model_name: HuggingFace model name (default: ProsusAI/finbert)
            device: Device to run model on ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_name = model_name
        self.device = self._get_device(device)

        logger.info(f"Loading FinBERT model on {self.device}...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode

            logger.info("FinBERT model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading FinBERT model: {e}")
            raise

        # Sentiment labels for FinBERT
        self.labels = {0: "Negative", 1: "Neutral", 2: "Positive"}

    def _get_device(self, device: Optional[str]) -> torch.device:
        """Determine the best available device"""
        if device is not None:
            return torch.device(device)

        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def analyze_text(self, text: str, max_length: int = 512) -> SentimentResult:
        """
        Analyze sentiment of a single text.

        Args:
            text: Input text to analyze
            max_length: Maximum sequence length for tokenization

        Returns:
            SentimentResult: Sentiment analysis results
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True
            ).to(self.device)

            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Get probabilities
            probs = predictions.cpu().numpy()[0]

            # Calculate overall sentiment score (Positive - Negative)
            sentiment_score = float(probs[2] - probs[0])

            # Get sentiment label
            predicted_label_id = probs.argmax()
            sentiment_label = self.labels[predicted_label_id.item()]

            return SentimentResult(
                sentiment_score=sentiment_score,
                sentiment_label=sentiment_label,
                positive_score=float(probs[2]),
                negative_score=float(probs[0]),
                neutral_score=float(probs[1])
            )

        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            raise

    def analyze_batch(self, texts: List[str], batch_size: int = 16) -> List[SentimentResult]:
        """
        Analyze sentiment for multiple texts (batch processing).

        Args:
            texts: List of input texts
            batch_size: Batch size for processing (default: 16 for GPU optimization)

        Returns:
            List[SentimentResult]: Sentiment analysis results for all texts
        """
        if not texts:
            return []

        results = []

        # Process in batches for GPU optimization
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            try:
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)

                # Run inference
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

                # Process results
                probs_batch = predictions.cpu().numpy()

                for probs in probs_batch:
                    sentiment_score = float(probs[2] - probs[0])
                    predicted_label_id = probs.argmax()
                    sentiment_label = self.labels[predicted_label_id.item()]

                    results.append(SentimentResult(
                        sentiment_score=sentiment_score,
                        sentiment_label=sentiment_label,
                        positive_score=float(probs[2]),
                        negative_score=float(probs[0]),
                        neutral_score=float(probs[1])
                    ))

            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size}: {e}")
                # Return neutral sentiment for failed texts
                for _ in batch_texts:
                    results.append(SentimentResult(
                        sentiment_score=0.0,
                        sentiment_label="Neutral",
                        positive_score=0.33,
                        negative_score=0.33,
                        neutral_score=0.34
                    ))

        return results


# Example usage and testing
if __name__ == "__main__":
    # Initialize model
    print("Initializing FinBERT model...")
    analyzer = FinBERTSentiment()

    # Test single text analysis
    print("\n=== Single Text Analysis ===")
    test_texts = [
        "The bank reported strong quarterly earnings with a 15% increase in net profit.",
        "Concerns about rising bad debts and worsening economic conditions.",
        "The stock price remained stable with minor fluctuations throughout the day.",
    ]

    for text in test_texts:
        result = analyzer.analyze_text(text)
        print(f"\nText: {text[:60]}...")
        print(f"Result: {result}")

    # Test batch analysis
    print("\n=== Batch Analysis ===")
    batch_results = analyzer.analyze_batch(test_texts)
    print(f"Processed {len(batch_results)} texts")
    for i, result in enumerate(batch_results):
        print(f"Text {i+1}: {result}")

    print("\n=== Analysis Complete ===")
