# QuASy Anonymizer PoC

This notebook demonstrates the core anonymization logic for QuASy (Query Anonymization System). QuASy enables queries with private trade secrets, inventions, or other forms of intellectual property to be part of a private language model, and all queries related to the aforementioned will be abstracted by QuASy to research public LLMs - and the delta of responses given feedback for will be added to private language model (PLM) for tuning.

**Goal**: Detect sensitive IP in a query and abstract it with placeholders (e.g., `[[PRODUCT_0]]`), creating a safe version for public LLMs while keeping a mapping for private reconstruction.

**Components:**
- Entity Detection: spaCy NER + custom regex.
- Abstraction: Replace with typed placeholders.
- Reconstruction: Reverse for PLM (stub).


```python
# Install dependencies (run once)
!pip install spacy
```


```python
# Download SpaCy model (run once)
!python -m spacy download en_core_web_trf
```


```python
# Anonymizer Code (Core Logic)
import re
from typing import Tuple, Dict, List

import spacy

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)["spacy"]
MODEL_NAME = config["model_name"]
nlp = spacy.load(MODEL_NAME)


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
```



## Test Case 1

Query with trade secrets (alloys as products)


```python
query = "Our new alloy X17 improves turbine efficiency by 12% vs. Inconel 718."

safe, mapping = anonymize(query)

print(f"Safe Query: {safe}")
print(f"Mapping: {mapping}")

full = reconstruct_query(safe, mapping)
print(f"Reconstructed: {reconstruct_query}")

assert full == query
```

## Test Case 2

Query with invention (patent) and formula

## Test Case 2

Query with invention (patent) and formula


```python
query = "Our new alloy X17 improves turbine efficiency by 12% vs. Inconel 718."

safe, mapping = anonymize(query)

print(f"Safe Query: {safe}")
print(f"Mapping: {mapping}")

full = reconstruct_query(safe, mapping)
print(f"Reconstructed: {reconstruct_query}")

assert full == query
```

## Test Case 3: Code Block + Person/Org

Query with code snippet and named entities


```python
query = "Elon Musk at Tesla said: ```def secret_algo(): return 42``` is out IP."

safe, mapping = anonymize(query)

print(f"Safe Query: {safe}")
print(f"Mapping: {mapping}")

full = reconstruct_query(safe, mapping)
print(f"Reconstructed: {reconstruct_query}")

assert full == query
```


```python

```

## Next Steps

- Integrate with public LLM: Send `safe` to OpenAI/Grok.
- PLM Baseline: Use `full` with local model.
- Delta Tuning: Compare responses (enterprise).

For full system, see QuASy blueprint.


```python

```
