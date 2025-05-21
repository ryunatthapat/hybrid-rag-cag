import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from utils.data_loader import load_company_faq
from dotenv import load_dotenv
import time

load_dotenv()

MODEL_NAME = os.getenv("HF_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.1")
HF_TOKEN = os.getenv("HF_TOKEN")
FAQ_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'company-faq.md'))

def get_device_and_dtype():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    else:
        return torch.device("cpu"), torch.float32

def load_model_and_tokenizer(model_name, hf_token):
    device, dtype = get_device_and_dtype()
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token
    )
    model.to(device)
    return tokenizer, model, device

def get_kv_cache(model, tokenizer, prompt: str) -> DynamicCache:
    device = model.model.embed_tokens.weight.device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    cache = DynamicCache()
    with torch.no_grad():
        _ = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
    return cache, input_ids.shape[-1]

def clean_up(cache: DynamicCache, origin_len: int):
    for i in range(len(cache.key_cache)):
        cache.key_cache[i] = cache.key_cache[i][:, :, :origin_len, :]
        cache.value_cache[i] = cache.value_cache[i][:, :, :origin_len, :]

def generate(model, input_ids, past_key_values, max_new_tokens=128):
    device = model.model.embed_tokens.weight.device
    origin_len = input_ids.shape[-1]
    input_ids = input_ids.to(device)
    output_ids = input_ids.clone()
    next_token = input_ids
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
            logits = out.logits[:, -1, :]
            token = torch.argmax(logits, dim=-1, keepdim=True)
            output_ids = torch.cat([output_ids, token], dim=-1)
            past_key_values = out.past_key_values
            next_token = token.to(device)
            if model.config.eos_token_id is not None and token.item() == model.config.eos_token_id:
                break
    return output_ids[:, origin_len:]

if __name__ == "__main__":
    print(f"Loading modellll: {MODEL_NAME}")
    tokenizer, model, device = load_model_and_tokenizer(MODEL_NAME, HF_TOKEN)
    print(f"Using device: {device}")
    print(f"Tokenizer name: {getattr(tokenizer, 'name_or_path', str(tokenizer))}")
    print(f"Model name: {getattr(model, 'name_or_path', str(model))}")

    print(f"Loading FAQ from: {FAQ_PATH}")
    faq_text = load_company_faq(FAQ_PATH)
    print(f"Loaded FAQ length: {len(faq_text)} characters")

    # --- Prepare knowledge prompt for Mistral ---
    system_prompt = f"""
<|system|>
You are an assistant who provides concise answers.
<|user|>
Context:
{faq_text}
Question:
""".strip()

    print("Building KV cache for FAQ...")
    t1 = time.time()
    kv_cache, origin_len = get_kv_cache(model, tokenizer, system_prompt)
    t2 = time.time()
    print(f"KV cache built. Length: {origin_len}. Time: {t2-t1:.2f}s")

    # --- Answer questions using the cache ---
    questions = [
        "what is PALO IT?",
        "How to contact PALO IT?",
        "what technical stacks PALO can offer?"
    ]
    for i, q in enumerate(questions, 1):
        clean_up(kv_cache, origin_len)
        print(f"\nQ{i}: {q}")
        input_ids = tokenizer(q + "\n", return_tensors="pt").input_ids.to(device)
        output_ids = generate(model, input_ids, kv_cache, max_new_tokens=300)
        answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"A{i}: {answer}")