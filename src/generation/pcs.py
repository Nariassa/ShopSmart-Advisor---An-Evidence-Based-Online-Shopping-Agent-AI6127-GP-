"""Transparent, experimental PCS; all components and missingness are returned."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
from typing import Mapping, Sequence

from .evidence import Evidence, canonical_aspect, clean_number
from .sentiment import Sentiment


@dataclass(frozen=True)
class PCSConfig:
    quality_weight: float = 0.40
    consistency_weight: float = 0.35
    recency_weight: float = 0.25
    volume_prior: float = 5.0
    half_life_days: float = 180.0
    unknown_recency: float = 0.5
    min_aspect_units: int = 2

    def __post_init__(self):
        weights = (self.quality_weight, self.consistency_weight, self.recency_weight)
        if any(not math.isfinite(w) or w < 0 for w in weights) or not math.isclose(sum(weights), 1):
            raise ValueError("PCS weights must be finite, non-negative and sum to one.")
        if not math.isfinite(self.volume_prior) or not math.isfinite(self.half_life_days):
            raise ValueError("PCS scale parameters must be finite.")
        if self.volume_prior <= 0 or self.half_life_days <= 0 or self.min_aspect_units < 1:
            raise ValueError("PCS scale parameters must be positive.")
        if not 0 <= self.unknown_recency <= 1:
            raise ValueError("unknown_recency must be in [0, 1].")


def unit_interval(value, name: str) -> float:
    number = clean_number(value)
    if number is None or not 0 <= number <= 1:
        raise ValueError(f"{name} must be a finite number in [0, 1].")
    return number


def review_time(value) -> datetime | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        if isinstance(value, datetime):
            parsed = value
        else:
            number = clean_number(value)
            if number is not None:
                parsed = datetime.fromtimestamp(number / 1000 if abs(number) > 1e11 else number, timezone.utc)
            else:
                parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    except (ValueError, TypeError, OverflowError, OSError):
        return None


def quality(item: Evidence) -> tuple[float, list[str]]:
    missing = []
    score = 0.5 + 0.1 * min(len(item.body.split()) / 100, 1.0)
    verified = item.metadata.get("verified_purchase", item.metadata.get("verified"))
    if verified in (True, "true", "True"):
        score += 0.2
    elif verified not in (False, "false", "False"):
        missing.append("verified_purchase")
    votes = clean_number(item.metadata.get("helpful_vote", item.metadata.get("helpful_votes")))
    if votes is None or votes < 0:
        missing.append("helpful_vote")
    else:
        score += 0.2 * min(math.log1p(votes) / math.log(101), 1.0)
    return min(score, 1.0), missing


def aspect_distribution(counts: Mapping) -> dict:
    values = {}
    for label in ("positive", "negative", "neutral"):
        number = clean_number(counts.get(label, 0))
        if number is None or number < 0 or number != int(number):
            raise ValueError("Aspect counts must be non-negative finite integers.")
        values[label] = int(number)
    total = sum(values.values())
    proportions = {key: count / total if total else 0.0 for key, count in values.items()}
    polarization = 4 * proportions["positive"] * proportions["negative"]
    return {**values, "total": total, "proportions": proportions,
            "polarization": polarization,
            "fit_score": proportions["positive"] + 0.5 * proportions["neutral"] if total else 0.5}


def calculate_pcs(
    evidence: Sequence[Evidence],
    sentiments: Mapping[str, Sentiment],
    requested_aspects: Sequence[str],
    *,
    aspect_weights: Mapping[str, float] | None = None,
    signals: Mapping | None = None,
    config: PCSConfig | None = None,
    now: datetime | None = None,
) -> dict:
    """Compute one product's score. Aspect-analysis signals use count data, not percentages."""
    config = config or PCSConfig()
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    if len({item.parent_asin for item in evidence}) > 1:
        raise ValueError("calculate_pcs accepts exactly one product at a time.")
    requested = list(dict.fromkeys(canonical_aspect(a) for a in requested_aspects)) or ["general"]
    raw_weights = {canonical_aspect(k): clean_number(v) for k, v in (aspect_weights or {}).items()}
    if any(value is None or value < 0 for value in raw_weights.values()):
        raise ValueError("Aspect weights must be finite non-negative numbers.")
    weights = {a: raw_weights.get(a, 1.0) for a in requested}
    weight_sum = sum(weights.values())
    if not weight_sum:
        raise ValueError("At least one requested aspect needs a positive weight.")
    weights = {a: value / weight_sum for a, value in weights.items()}
    reviews = [item for item in evidence if item.chunk_type == "review"]
    unique = {item.unit_id: item for item in reviews}
    counts: dict[str, dict[str, set]] = {}
    for item in reviews:
        label = sentiments[item.chunk_id].label
        bucket = counts.setdefault(item.aspect, {k: set() for k in ("positive", "negative", "neutral")})
        bucket[label].add(item.unit_id)
    # A review that contributes both polarities to the same aspect counts as neutral.
    distributions = {}
    for aspect, labels in counts.items():
        by_unit = {}
        for label, units in labels.items():
            for unit in units:
                by_unit.setdefault(unit, set()).add(label)
        final = {k: 0 for k in ("positive", "negative", "neutral")}
        for labels_for_unit in by_unit.values():
            label = next(iter(labels_for_unit)) if len(labels_for_unit) == 1 else "neutral"
            final[label] += 1
        distributions[aspect] = aspect_distribution(final)
    signals = signals or {}
    for aspect, aspect_counts in signals.get("aspects", {}).items():
        distributions[canonical_aspect(aspect)] = aspect_distribution(aspect_counts)
    if "general" in requested and "general" not in distributions:
        combined = {k: 0 for k in ("positive", "negative", "neutral")}
        for item in unique.values():
            combined[sentiments[item.chunk_id].label] += 1
        distributions["general"] = aspect_distribution(combined)
    covered = {a: min(distributions.get(a, {}).get("total", 0) / config.min_aspect_units, 1.0) for a in requested}
    coverage = sum(weights[a] * covered[a] for a in requested)
    polarization = sum(weights[a] * distributions.get(a, {}).get("polarization", 0) for a in requested)
    fit = sum(weights[a] * distributions.get(a, {}).get("fit_score", 0.5) for a in requested)
    if "coverage" in signals:
        coverage = unit_interval(signals["coverage"], "coverage")
    if "polarization" in signals:
        polarization = unit_interval(signals["polarization"], "polarization")
    missing = {"verified_purchase": 0, "helpful_vote": 0, "timestamp": 0, "source_review_id": 0}
    qualities = []
    recencies = []
    for item in unique.values():
        item_quality, unknown = quality(item)
        qualities.append(item_quality)
        for field in unknown:
            missing[field] += 1
        if not item.metadata.get("source_review_id"):
            missing["source_review_id"] += 1
        parsed = review_time(item.metadata.get("timestamp"))
        if parsed is None or parsed > now:
            missing["timestamp"] += 1
            recencies.append(config.unknown_recency)
        else:
            days = (now - parsed).total_seconds() / 86400
            recencies.append(0.5 ** (days / config.half_life_days))
    n = len(unique)
    q = sum(qualities) / n if n else 0.0
    r = sum(recencies) / n if n else config.unknown_recency
    v = n / (n + config.volume_prior)
    consistency = 1 - polarization
    base = config.quality_weight * q + config.consistency_weight * consistency + config.recency_weight * r
    score = 100 * v * coverage * base if n else 0.0
    return {
        "score": round(score, 2), "fit_score": round(fit, 4),
        "components": {"quality": q, "consistency": consistency, "recency": r,
                       "volume": v, "coverage": coverage, "polarization": polarization},
        "review_evidence_units": n, "review_chunks": len(reviews),
        "requested_aspects": requested, "aspect_weights": weights,
        "coverage_by_aspect": covered, "aspect_sentiments": distributions,
        "missing_metadata": missing, "config": asdict(config),
        "signal_source": "member_d" if signals else "retrieved_evidence",
        "sentiment_methods": sorted({sentiments[item.chunk_id].method for item in reviews}),
        "scope": "retrieved_sample",
        "interpretation": "Experimental evidence confidence, not purchase success probability.",
    }
