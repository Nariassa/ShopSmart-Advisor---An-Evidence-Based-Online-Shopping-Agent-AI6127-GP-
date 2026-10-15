"""Evidence facts and fail-closed claim validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence
import unicodedata
import re
from html import unescape

from .evidence import ASPECT_TERMS, Evidence, canonical_aspect, sentence_spans
from .sentiment import Sentiment


@dataclass(frozen=True)
class Fact:
    fact_id: str
    source_id: str
    chunk_id: str
    parent_asin: str
    chunk_type: str
    aspect: str
    quote: str
    start: int
    end: int
    sentiment: str
    relevance: float

    def to_dict(self) -> dict:
        return asdict(self)


def extract_facts(
    evidence: Sequence[Evidence], query: str, requested_aspects: Sequence[str],
    classifier, sentiments: Mapping[str, Sentiment], max_facts: int = 60,
    max_context_chars: int = 16000,
) -> list[Fact]:
    import re
    tokens = set(re.findall(r"[a-z]{3,}", query.lower()))
    tokens -= {"the", "this", "that", "these", "those", "for", "with", "and",
               "are", "was", "have", "has", "can", "could", "should", "would",
               "how", "which", "what", "does", "need", "find", "want", "from"}
    requested = {canonical_aspect(a) for a in requested_aspects}
    candidates = []
    for source_index, item in enumerate(evidence, 1):
        for quote, start, end in sentence_spans(item):
            words = set(re.findall(r"[a-z]{3,}", quote.lower()))
            overlap = len(tokens & words) / max(len(tokens), 1)
            sentiment = classifier.classify(quote).label if item.chunk_type == "review" else "neutral"
            # For a specific query, facts need an aspect tag or a query/aspect term.
            aspect_match = item.aspect in requested
            keyword_match = any(term in quote.lower() for aspect in requested
                                for term in ASPECT_TERMS.get(aspect, ()))
            if "general" not in requested and not (keyword_match or overlap or
                                                      (aspect_match and sentiment != "neutral")):
                continue
            if sentiments[item.chunk_id].method == "provided_absa":
                sentiment = sentiments[item.chunk_id].label
            relevance = overlap + 0.5 * aspect_match + 0.25 * keyword_match
            candidates.append(Fact("", f"E{source_index}", item.chunk_id, item.parent_asin,
                                   item.chunk_type, item.aspect, quote, start, end,
                                   sentiment, relevance))
    candidates.sort(key=lambda fact: (-fact.relevance, fact.chunk_id, fact.start))
    # Round-robin product/polarity groups so one product's positive reviews
    # cannot fill the entire context while opposing evidence is available.
    groups = {}
    for fact in candidates:
        groups.setdefault((fact.parent_asin, fact.chunk_type, fact.sentiment), []).append(fact)
    result = []
    consumed = 0
    specs_per_product = {}
    while any(groups.values()) and len(result) < max_facts:
        for group in groups.values():
            if not group or len(result) >= max_facts:
                continue
            fact = group.pop(0)
            if fact.chunk_type == "spec":
                if specs_per_product.get(fact.parent_asin, 0) >= 3:
                    continue
            # Count metadata overhead conservatively as well as quote text.
            size = len(fact.quote) + 300
            if consumed + size > max_context_chars:
                continue
            consumed += size
            if fact.chunk_type == "spec":
                specs_per_product[fact.parent_asin] = specs_per_product.get(fact.parent_asin, 0) + 1
            result.append(Fact(**{**fact.to_dict(), "fact_id": f"F{len(result)+1}"}))
    return result


class NLIVerifier:
    """Optional paraphrase check. Exact quotation needs no model."""

    def __init__(self, model: str = "cross-encoder/nli-deberta-v3-small", threshold: float = 0.9):
        if not 0 < threshold <= 1:
            raise ValueError("NLI threshold must be in (0, 1].")
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model).eval()
        mapping = {str(label).lower(): int(index) for index, label in self.model.config.id2label.items()}
        if "entailment" not in mapping:
            # This published model uses 0=contradiction, 1=entailment, 2=neutral.
            if model == "cross-encoder/nli-deberta-v3-small" and self.model.config.num_labels == 3:
                mapping = {"contradiction": 0, "entailment": 1, "neutral": 2}
            else:
                raise ValueError("NLI model must expose a known entailment label.")
        self.entailment_index = mapping["entailment"]
        self.label_mapping = mapping
        self.model_name = model
        self.threshold = threshold

    def score(self, premise: str, claim: str) -> dict:
        import torch
        inputs = self.tokenizer(premise, claim, return_tensors="pt", truncation=False)
        maximum = min(self.tokenizer.model_max_length, self.model.config.max_position_embeddings)
        if inputs["input_ids"].shape[1] > maximum:
            return {"supported": False, "entailment": 0.0, "reason": "input_too_long"}
        with torch.inference_mode():
            probabilities = self.model(**inputs).logits.softmax(-1)[0]
        scores = {name: float(probabilities[index]) for name, index in self.label_mapping.items()}
        return {"supported": scores["entailment"] >= self.threshold, **scores}

    def supports(self, premise: str, claim: str) -> tuple[bool, float]:
        result = self.score(premise, claim)
        return result["supported"], result["entailment"]


def uses_latin_letters(text):
    """Reject Chinese/other scripts; prompt enforces English, this is not a language detector."""
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and all("LATIN" in unicodedata.name(c, "") for c in letters)


def canonical_product_name(text):
    # Presentation differences only: never discard words, model numbers or suffixes.
    return ' '.join(unicodedata.normalize('NFC', unescape(text)).split())


def rewrite_safety(premise, text, review=True):
    """Conservative necessary checks, supplementary to NLI (not semantic proof)."""
    p, t = unescape(premise).lower(), unescape(text).lower()
    if review and not t.startswith('one reviewer reports that '):
        return 'review_attribution_missing'
    if re.search(r'\b(?:i|my|me|we|our|us)\b', p):
        content = t.removeprefix('one reviewer reports that ')
        if not re.search(r"\b(?:the reviewer|their|they)\b", content):
            return 'personal_scope_lost'
    if re.search(r'\b(?:everyone|all users|always|guaranteed|never fails)\b', t) and not re.search(r'\b(?:everyone|all users|always|guaranteed|never fails)\b', p):
        return 'scope_strengthened'
    numbers = lambda s: re.findall(r'\d+(?:[.,]\d+)?\s*(?:\+|%|hours?|hrs?|minutes?|mins?|days?|weeks?|months?|years?|mah|wh|watts?|w|gb|tb|mm|cm|inches|inch)?', s)
    numeric = lambda s: sorted(re.sub(r'\s+', '', n).rstrip('.') for n in numbers(s))
    if numeric(p) != numeric(t):
        return 'numeric_detail_changed'
    # Preserve explicit logical operators; prefer safe quote fallback over risky compression.
    negative = r"\b(?:not|no|never|without|cannot|can't|doesn't|didn't|isn't|wasn't|won't|wouldn't|don't)\b|n['’]t\b"
    if bool(re.search(negative, p)) != bool(re.search(negative, t)):
        return 'negation_changed'
    for token in ['if','unless','only','when','until','after','before','during','sometimes','often','usually','typically','perhaps','maybe','might','may','could','so far','at least','at most','up to']:
        pattern = r'(?<!\w)' + re.escape(token) + r'(?!\w)'
        if re.search(pattern, p) and not re.search(pattern, t):
            return 'condition_or_uncertainty_lost'
    return None


def verify_selections(payload, facts, evidence, max_claims=24):
    """Bind IDs and exact canonical product titles; hydrate quotes server-side."""
    if not isinstance(payload, Mapping) or set(payload) != {"selections"} or not isinstance(payload["selections"], list):
        raise ValueError("Selection response must contain only a selections array.")
    if len(payload["selections"]) > max_claims:
        raise ValueError("Selection response exceeds the configured limit.")
    by_fact = {fact.fact_id: fact for fact in facts}
    by_chunk = {item.chunk_id: item for item in evidence}
    accepted, rejected, seen = [], [], set()
    for selection in payload["selections"]:
        if not isinstance(selection, Mapping) or set(selection) != {"fact_id", "product_name"}:
            rejected.append({"selection": selection, "reason": "invalid_selection_fields"})
            continue
        identity = selection["fact_id"]
        fact = by_fact.get(identity) if isinstance(identity, str) else None
        source = by_chunk.get(fact.chunk_id) if fact else None
        if fact is None or source is None:
            rejected.append({"selection": dict(selection), "reason": "unknown_fact"})
            continue
        if not isinstance(selection['product_name'], str) or canonical_product_name(selection["product_name"]) != canonical_product_name(source.title):
            rejected.append({"selection": dict(selection), "reason": "product_name_mismatch"})
            continue
        if identity in seen:
            rejected.append({"selection": dict(selection), "reason": "duplicate_fact"})
            continue
        claims, failures = verify_claims({"claims": [{"fact_id": identity, "text": fact.quote}]},
                                         facts, evidence, max_claims=1, english_only=True)
        rejected.extend(failures)
        if claims:
            seen.add(identity)
            accepted.extend(claims)
    return accepted, rejected


def verify_claims(payload: Mapping, facts: Sequence[Fact], evidence: Sequence[Evidence],
                  verifier=None, max_claims: int = 24, *, english_only=False, strict_rewrite=False) -> tuple[list[dict], list[dict]]:
    if not isinstance(payload, Mapping) or not isinstance(payload.get("claims"), list):
        raise ValueError("Generator response must contain a claims array.")
    if strict_rewrite and len(payload['claims']) > max_claims:
        raise ValueError('Rewrite response exceeds the configured limit.')
    by_fact = {fact.fact_id: fact for fact in facts}
    by_chunk = {item.chunk_id: item for item in evidence}
    accepted, rejected = [], []
    seen = set()
    for proposed in payload["claims"][:max_claims]:
        if not isinstance(proposed, Mapping):
            rejected.append({"reason": "invalid_claim_object"})
            continue
        proposed_id = proposed.get("fact_id")
        fact = by_fact.get(proposed_id) if isinstance(proposed_id, str) else None
        text = proposed.get("text")
        if fact is None or not isinstance(text, str) or not text.strip():
            rejected.append({"claim": dict(proposed), "reason": "unknown_fact_or_empty_text"})
            continue
        source = by_chunk.get(fact.chunk_id)
        if (source is None or source.parent_asin != fact.parent_asin
                or source.text[fact.start:fact.end] != fact.quote):
            rejected.append({"claim": dict(proposed), "reason": "source_offset_mismatch"})
            continue
        text = text.strip()
        if english_only and not uses_latin_letters(text):
            rejected.append({"claim": dict(proposed), "reason": "non_english_script"})
            continue
        if len(text) > 1000:
            rejected.append({"claim": dict(proposed), "reason": "claim_too_long"})
            continue
        method, support_score = "exact_quote", 1.0
        if text != fact.quote:
            if verifier is None:
                rejected.append({"claim": dict(proposed), "reason": "paraphrase_requires_nli"})
                continue
            if english_only and not uses_latin_letters(fact.quote):
                rejected.append({"claim": dict(proposed), "reason": "nli_source_non_english"})
                continue
            if strict_rewrite:
                failure = rewrite_safety(fact.quote, text, fact.chunk_type == 'review')
                if failure:
                    rejected.append({'claim':dict(proposed), 'reason':failure, 'stage':'rewrite_guard'})
                    continue
            # Attribution is checked independently. NLI compares the stated content,
            # not a new reporting event absent from a first-person review sentence.
            nli_text = text
            if strict_rewrite and fact.chunk_type == 'review':
                nli_text = text[len('One reviewer reports that '):]
            supported, support_score = verifier.supports(fact.quote, nli_text)
            if not supported:
                rejected.append({"claim": dict(proposed), "reason": "nli_not_entailed",
                                 "support_score": support_score})
                continue
            method = "nli_entailment"
        identity = (fact.parent_asin, text)
        if identity in seen:
            continue
        seen.add(identity)
        kind = "pro" if fact.sentiment == "positive" else "con" if fact.sentiment == "negative" else "finding"
        accepted.append({"text": text, "parent_asin": fact.parent_asin, "kind": kind,
                         "product_name": source.title,
                         "fact_id": fact.fact_id, "grounding_method": method,
                         "support_score": support_score,
                         "citations": [{"source_id": fact.source_id, "chunk_id": fact.chunk_id,
                                        "chunk_type": fact.chunk_type, "quote": fact.quote,
                                        "start": fact.start, "end": fact.end}]})
    return accepted, rejected
