"""Presentation-only handling of evidence citations."""
import re


def normalize_citation_labels(text, citations):
    """Map a bare numeric reference only when its source exists."""
    def replace(match):
        source = f'C{int(match.group(1))}'
        return f'[{source}]' if source in citations else match.group(0)
    return re.sub(r'\[(\d+)\]', replace, text)
