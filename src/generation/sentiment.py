"""Replaceable sentiment baseline; product-level stars are never ABSA."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re

from .evidence import Evidence, clean_number


@dataclass(frozen=True)
class Sentiment:
    label: str
    score: float
    confidence: float
    method: str

    def to_dict(self) -> dict:
        return asdict(self)


class LexiconSentiment:
    """Small, disclosed English fallback, with local negation handling."""

    positive = {"good", "great", "excellent", "comfortable", "reliable",
                "durable", "clear", "amazing", "love", "easy", "stable",
                "perfect", "impressive", "helpful", "worth", "convenient"}
    negative = {"bad", "poor", "terrible", "uncomfortable", "unreliable",
                "broken", "disappointing", "disappointed", "hate", "waste",
                "defective", "useless", "distortion", "disconnects", "failed"}
    negators = {"not", "never", "no", "hardly", "isn't", "wasn't",
                "don't", "doesn't", "cannot", "can't"}

    def classify(self, text: str) -> Sentiment:
        # Replace a few explicit failure phrases before local token scoring.
        # This remains a baseline, not a validated aspect sentiment model.
        normalized = text.lower()
        failures = (r"(?:can(?:not|'t)|could(?: not|n't)|does(?: not|n't)|do(?: not|n't)|will(?: not|n't))\s+(?:\w+\s+){0,2}(?:connect|work)",
                    r"(?:no longer|stopped)\s+(?:connect(?:ing)?|work(?:ing)?)")
        for pattern in failures:
            normalized = re.sub(pattern, " defective ", normalized)
        tokens = re.findall(r"[a-z]+(?:'[a-z]+)?|[.,;:!?]", normalized)
        positives = negatives = 0
        scope = []
        for token in tokens:
            if token in {".", ",", ";", ":", "!", "?", "but", "however"}:
                scope = []
                continue
            polarity = 1 if token in self.positive else -1 if token in self.negative else 0
            if polarity and any(word in self.negators for word in scope[-8:]):
                polarity *= -1
            positives += polarity > 0
            negatives += polarity < 0
            scope.append(token)
        total = positives + negatives
        score = (positives - negatives) / total if total else 0.0
        label = "positive" if score > 0.15 else "negative" if score < -0.15 else "neutral"
        return Sentiment(label, score, 0.55 if total else 0.0, "lexicon_baseline")


class TransformerSentiment:
    """Optional classifier: instantiated only when explicitly requested."""

    def __init__(self, model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        from transformers import pipeline
        self.pipeline = pipeline("text-classification", model=model, device=-1)

    def classify(self, text: str) -> Sentiment:
        output = self.pipeline(text, truncation=True, top_k=None)
        if output and isinstance(output[0], list):
            output = output[0]
        labels = {str(item["label"]).lower(): float(item["score"]) for item in output}
        if not {"positive", "negative", "neutral"} <= labels.keys():
            raise ValueError("Sentiment model must expose positive/negative/neutral labels.")
        label = max(labels, key=labels.get)
        return Sentiment(label, labels["positive"] - labels["negative"],
                         labels[label], "transformer")


def evidence_sentiment(item: Evidence, classifier) -> Sentiment:
    if item.chunk_type != "review":
        return Sentiment("neutral", 0.0, 0.0, "spec_not_review")
    label = str(item.metadata.get("sentiment_label", item.metadata.get("sentiment", ""))).lower()
    score = clean_number(item.metadata.get("sentiment_score"))
    if label in {"positive", "negative", "neutral"}:
        mapped = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}[label]
        if score is not None and not -1 <= score <= 1:
            raise ValueError("Aspect Analysis & Purchase Decision Explanation sentiment_score must be in [-1, 1].")
        return Sentiment(label, mapped if score is None else score, 1.0, "provided_absa")
    if score is not None:
        if not -1 <= score <= 1:
            raise ValueError("Aspect Analysis & Purchase Decision Explanation sentiment_score must be in [-1, 1].")
        label = "positive" if score > 0.15 else "negative" if score < -0.15 else "neutral"
        return Sentiment(label, score, 1.0, "provided_absa")
    return classifier.classify(item.body)
