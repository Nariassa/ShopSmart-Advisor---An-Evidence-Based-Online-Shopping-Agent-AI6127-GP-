"""Conservative, auditable title-based product-type gate; not a semantic classifier."""
import re
from html import unescape


# Specific requested accessories precede the devices they mention.
TYPES = [
    ('tablet case', r'(?:tablet|ipad) case', r'\b(?:case|cover|folio)\b', r'\b(?:tablet|ipad|galaxy tab|kindle|surface)\b'),
    ('phone case', r'phone case', r'\b(?:case|cover)\b', r'\b(?:phone|iphone|galaxy|pixel)\b'),
    ('camera battery', r'camera batter', r'\bbatter(?:y|ies)\b', r'\b(?:camera|canon|nikon|sony|panasonic|olympus|fujifilm|gopro|pentax)\b'),
    ('camera bag', r'camera bag', r'\b(?:bag|backpack|case|sling)\b', r'\b(?:camera|dslr|photograph)\w*'),
    ('laptop charger', r'laptop charger', r'\b(?:charger|ac adapter|power adapter|power supply)\b', r'\b(?:laptop|notebook|macbook|thinkpad|chromebook|inspiron|latitude|pavilion|presario|satellite|acer|asus|lenovo|dell|hp|surface)\b'),
    ('bluetooth adapter', r'bluetooth adapter', r'\b(?:adapter|adaptor|dongle|receiver|transmitter)\b', r'\bbluetooth\b'),
    ('cooling fan', r'cooling fan', r'\bfan\b', r'\b(?:cooling|computer|pc|desktop|case|cpu|chassis)\b'),
    ('earbuds', r'earbuds', r'\b(?:earbuds?|earphones?|in[ -]ear|airpods|buds)\b', None),
    ('headphones', r'headphones', r'\b(?:headphones?|headsets?|earbuds?|earphones?)\b', None),
    ('speaker', r'\bspeaker', r'\bspeakers?\b', None),
    ('monitor', r'\bmonitor\b', r'\bmonitor\b', None),
    ('laptop', r'\blaptop\b', r'\b(?:laptop|notebook|chromebook|macbook|thinkpad|ultrabook)\b', None),
    ('tablet', r'\btablet\b', r'\b(?:tablet|ipad|kindle|galaxy tab)\b', None),
    ('security camera', r'security camera', r'\b(?:camera|cameras|surveillance|cctv)\b', None),
    ('digital camera', r'digital camera', r'\b(?:camera|dslr|coolpix|powershot|cyber[ -]?shot)\b', None),
    ('smartwatch', r'\bsmartwatch\b', r'\b(?:smartwatch|smart watch|watch|fitness tracker|fitbit)\b', None),
]
ACCESSORY = r'\b(?:case|cover|bag|charger|adapter|adaptor|battery|batteries|cable|mount|stand|protector|replacement|earpads?|cushions?|screen replacement)\b'


def requirements(query, aspects, supplied):
    output = dict(supplied)
    selected = next((name for name, pattern, _, _ in TYPES if re.search(pattern, query, re.I)), None)
    output.setdefault('target_product_type', selected)
    output.setdefault('required_conditions', [query.strip()])
    output.setdefault('explicit_condition_phrases', [m.group(0) for m in re.finditer(
        r'\b(?:long battery life|stable connection|clear microphone quality|strong bass|accurate colors|sharp display|easy to (?:set up|use|install)|reliable connectivity|lightweight|video editing|multitasking|quiet|durable|not bulky|low light|focuses quickly|long runtime|comfortable to carry|drop protection|water resistance|without (?:distortion|frequent dropouts|overheating)|responsive warranty support|straightforward replacements)\b', query, re.I)])
    output.setdefault('exclusions', ['Different product types; accessories when a device is requested'] if selected else [])
    output['explicit_exclusion_phrases'] = [m.group(0) for m in re.finditer(r'\b(?:without|not)\s+[^,?.;]+', query, re.I)]
    output['requested_aspects'] = list(aspects)
    output['condition_policy'] = 'Missing feature evidence is unverified, not satisfied; title matching establishes type only.'
    return output


def filter_products(evidence, constraints):
    target = constraints.get('target_product_type')
    definition = next((d for d in TYPES if d[0] == target), None)
    if definition is None:
        return list(evidence), []
    _, _, required, context = definition
    decisions = {}
    for item in evidence:
        if item.parent_asin in decisions:
            continue
        title = unescape(item.title).lower()
        # Compatibility mentions do not establish the sold item's type.
        head = re.split(r'\b(?:for|compatible with|fits)\b', title, maxsplit=1)[0]
        head = re.sub(r'\bbattery (?:life|time)\b', 'runtime', head)
        matched = bool(re.search(required, head) and (context is None or re.search(context, title)))
        if target in {'earbuds','headphones','speaker','monitor','laptop','tablet','security camera','digital camera','smartwatch'}:
            matched = matched and not bool(re.search(ACCESSORY, head))
        # "earbuds with charging case" is a device bundle, not a standalone case.
        if target in {'earbuds','headphones'}:
            device_head = re.split(r'\bwith\b', head, maxsplit=1)[0]
            matched = bool(re.search(required, device_head) and not re.search(ACCESSORY, device_head))
            if 'bluetooth' in ' '.join(constraints.get('required_conditions',[])).lower():
                matched = matched and bool(re.search(r'\b(?:bluetooth|wireless)\b', title))
        if target == 'speaker' and 'portable' in ' '.join(constraints.get('required_conditions',[])).lower():
            matched = matched and not bool(re.search(r'\b(?:car|truck|coaxial|subwoofer|tweeter|ceiling|in-wall)\w*\b', title))
        decisions[item.parent_asin] = {'parent_asin':item.parent_asin, 'title':item.title,
            'target_product_type':target, 'accepted':matched,
            'reason':'title_type_match' if matched else 'type_mismatch_or_unverified',
            'method':'conservative_title_rules'}
    excluded = [d for d in decisions.values() if not d['accepted']]
    return [e for e in evidence if decisions[e.parent_asin]['accepted']], excluded
