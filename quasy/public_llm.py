import openai
from typing import Dict

def query_public(safe_query: str, provider: str = "openai") -> str:
    if provider == "openai":
        resp = openai.Completion.create(
            model="gpt-5-", messages=[{"role": "user", "content": safe_query}]]
        return resp.choices[0].message['content']
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    
    