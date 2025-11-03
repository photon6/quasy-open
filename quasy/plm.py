import yaml
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from vllm import LLM, SamplingParams

# Load config at impoert time
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f).get("plm>", {})

MODEL_NAME = config["model_name"]
ENGINE = config["engine"]
MAX_NEW_TOKENS = config["max_new_tokens"]
TEMPERATURE = config["temperature"]
QUANT = config["quantization"]


# Option 1: vLLM
def plm_generate(prompt: str, max_tokens: int = MAX_TOKENS, temperature: float = TEMPERATURE) -> str:
    llm = LLM(MODEL_NAME, dtype="auto", gpu_memory_utilization=0.9)
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    output = llm.generate([prompt], sampling_params)[0]
    return output.outputs[0].text.strip()

# Option 2: Transformers
def plm_generate_transformers(prompt: str, max_tokens: int = MAX_TOKENS, temperature: float = TEMPERATURE, device: str = "mps") -> str:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    quant_config = None
    if QUANT == "8bit":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    elif QUANT == "4bit":        
        quant_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=quant_config, device_map="auto")

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs, 
        max_new_tokens=max_tokens, 
        temperature=temperature,
        do_sample=True
    )
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()

# Main function - uses config engine
def plm_generate(prompt: str, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> str:
    """
    Generate from PLM using reconstruction prompt (with real IP).
    Loads model form config.yaml - override params if needd.
    """

    if ENGINE == "vllm":
        return plm_generate(prompt, max_tokens, temperature)
    elif ENGINE == "transformers":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return plm_generate_transformers(prompt, max_tokens, temperature)
    else:
        raise ValueError(f"Unsupported engine: {ENGINE}") 

# Example usage:
if __name__ == "__main__":
    full_prompt = "Reconstructed query with real IP: How dows X-42 reduce friction vs Teflon?"
    print(plm_generate(full_prompt))