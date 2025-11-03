import os
import openai
from typing import Optional

from openai import OpenAI

# Load API key from env (secure)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

client = OpenAI(api_key=api_key)

def query_public(
        safe_query: str, 
        model: str = "gpt-5",
        max_tokens: int = 512,
        temperature: float = 0.7,
                 ) -> str:
    
    """
    Query a public LLM with the safe (abstracted) query.
    Returns the response text for delta comparison with PLM.
    """
    try: 
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": safe_query}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"OpenAI API error: {e}")
    
# Example usage:
if __name__ == "__main__":
    safe = "Our new alloy [[PRODUCT_0]] improves turbine efficiency by 12% vs. [[PRODUCT_1]]."
    print(query_public(safe))    
    