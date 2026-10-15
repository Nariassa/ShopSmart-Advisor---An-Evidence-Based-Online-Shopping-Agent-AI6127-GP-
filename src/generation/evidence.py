"""Normalize the knowledge-base and retrieval modules' evidence without loading the large indexes."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import re
from typing import Iterable, Mapping


ASPECT_TERMS = {
    "price/value": ("price", "budget", "value", "cost", "affordable", "价格", "预算", "性价比"),
    "build quality": ("durable", "reliable", "reliability", "build quality", "broken", "耐用", "质量"),
    "specifications/performance": ("performance", "processor", "gaming", "fast", "性能", "处理器"),
    "battery life": ("battery", "charging", "charge", "续航", "电池"),
    "screen/display": ("screen", "display", "monitor", "colors", "屏幕", "显示"),
    "connectivity": ("bluetooth", "wifi", "wi-fi", "connect", "connection", "connectivity", "蓝牙", "连接"),
    "customer service": ("customer service", "warranty", "support", "售后", "保修"),
    "ease of use": ("beginner", "set up", "setup", "easy", "ease of use", "新手", "易用"),
    "sound quality": ("sound", "audio", "bass", "microphone", "noise", "音质", "降噪", "麦克风"),
    "design/appearance": ("design", "appearance", "lightweight", "portable", "外观", "轻便"),
    "comfort": ("comfort", "comfortable", "cushion", "fit", "舒适", "佩戴"),
}
ASPECT_ALIASES = {
    "battery": "battery life", "battery_life": "battery life",
    "sound": "sound quality", "audio": "sound quality",
    "price": "price/value", "value": "price/value",
    "build_quality": "build quality", "performance": "specifications/performance",
    "display": "screen/display", "screen": "screen/display",
    "ease_of_use": "ease of use", "customer_service": "customer service",
}
HEADER = re.compile(r"^\[Product:\s*(.*?)\s*\|\s*ASIN:\s*([^\]]+)\]\s*", re.DOTALL)


def canonical_aspect(value: str) -> str:
    value = str(value).strip().lower()
    return ASPECT_ALIASES.get(value, value or "general")


def detect_aspects(query: str) -> list[str]:
    lowered = query.lower()
    found = []
    for aspect, terms in ASPECT_TERMS.items():
        if any(re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", lowered)
               if term.isascii() else term in lowered for term in terms):
            found.append(aspect)
    return found or ["general"]


def clean_number(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class Evidence:
    chunk_id: str
    parent_asin: str
    chunk_type: str
    aspect: str
    text: str
    title: str
    body: str
    body_offset: int
    metadata: dict = field(default_factory=dict)

    @property
    def unit_id(self) -> str:
        """Prefer true review IDs; otherwise deduplicate identical text."""
        review_id = self.metadata.get("source_review_id")
        if review_id is not None and str(review_id).strip():
            return "review:" + str(review_id)
        normalized = " ".join(self.body.lower().split())
        return "text:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        return {**self.metadata, "chunk_id": self.chunk_id,
                "parent_asin": self.parent_asin, "chunk_type": self.chunk_type,
                "aspect": self.aspect, "text": self.text, "title": self.title}


def normalize_evidence(records: Iterable[Mapping]) -> list[Evidence]:
    result = []
    seen: dict[str, Evidence] = {}
    for raw in records:
        if hasattr(raw, "to_dict"):
            raw = raw.to_dict()
        row = {**dict(raw.get("metadata") or {}), **dict(raw)}
        chunk_id = row.get("chunk_id")
        asin = row.get("parent_asin", row.get("product_id"))
        text = row.get("text")
        if not chunk_id or not asin or not isinstance(text, str) or not text.strip():
            raise ValueError("Evidence needs chunk_id, parent_asin and non-empty text.")
        kind = str(row.get("chunk_type", "review")).lower()
        if kind not in {"spec", "review"}:
            raise ValueError(f"Unsupported chunk_type: {kind}")
        header = HEADER.match(text)
        if header and header.group(2).strip() != str(asin):
            raise ValueError(f"ASIN in text does not match metadata: {chunk_id}")
        offset = header.end() if header else 0
        title = str(row.get("title") or (header.group(1) if header else asin))
        item = Evidence(str(chunk_id), str(asin), kind,
                        canonical_aspect(row.get("aspect", "general")),
                        text, title, text[offset:], offset, row)
        if item.chunk_id in seen:
            if item != seen[item.chunk_id]:
                raise ValueError(f"Conflicting duplicate chunk_id: {item.chunk_id}")
            continue
        seen[item.chunk_id] = item
        result.append(item)
    return result


def sentence_spans(item: Evidence, max_chars: int = 600) -> list[tuple[str, int, int]]:
    """Keep complete sentences and offsets; never truncate a negated claim."""
    spans = []
    pattern = r"[^\n]+?(?:[!?](?=\s|[A-Z]|$)|\.(?=\s|$)|(?<=[a-zA-Z])\.(?=[A-Z])|$)"
    for match in re.finditer(pattern, item.body):
        original = match.group(0)
        quote = original.strip()
        if not quote or len(quote) > max_chars:
            continue
        start = item.body_offset + match.start() + len(original) - len(original.lstrip())
        spans.append((quote, start, start + len(quote)))
    return spans
