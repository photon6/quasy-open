# quasy/anonymizer.py
import re
from typing import Tuple, Dict, List

import spacy

# Load the transformer model once at import time
nlp = spacy.load("en_core_web_trf")


def detect_ip(text: str) -> List[Tuple[int, int, str]]:
    """
    Scan *text* for sensitive entities and return a list of
    ``(start, end, label)`` tuples.

    - NER: ORG, PRODUCT, PERSON, LAW
    - Regex: US patents, chemical formulas, code blocks
    """
    doc = nlp(text)
    spans: List[Tuple[int, int, str]] = []

    # ---------- NER ----------
    for ent in doc.ents:
        if ent.label_ in {"ORG", "PRODUCT", "PERSON", "LAW"}:
            spans.append((ent.start_char, ent.end_char, ent.label_))

    # ---------- REGEX ----------
    patterns = {
        "PATENT": r"\bUS\d{7,8}\b",                     # US1234567 or US12345678
        "FORMULA": r"\b[A-Z][a-z]?\d*(?:\([^)]*\))?\b", # H2O, NaCl, C6H12O6
        "CODE": r"```[\s\S]*?```",                     # ```...```
    }

    for label, pat in patterns.items():
        for m in re.finditer(pat, text):
            spans.append((m.start(), m.end(), label))

    # Sort by start offset (important for non‑overlapping replacement)
    return sorted(spans, key=lambda x: x[0])


def anonymize(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Replace every sensitive span with a placeholder ``[[LABEL_N]]``
    and return ``(safe_text, placeholder → original)``.
    """
    spans = detect_ip(text)
    mapping: Dict[str, str] = {}
    safe = ""
    offset = 0

    for i, (start, end, label) in enumerate(spans):
        placeholder = f"[[{label}_{i}]]"
        real = text[start:end]
        mapping[placeholder] = real
        safe += text[offset:start] + placeholder
        offset = end

    safe += text[offset:]
    return safe, mapping

def reconstruct_query(safe_query: str, mapping: Dict[str, str]) -> str:
    """
    Reconstruct the full quwery by replacing placeholders with original IP.
    Used for secure PLM basweline generation without public exposure.
    """
    
    for placeholder, real in mapping.items():
        safe_query = safe_query.replace(placeholder, real)
    return safe_query