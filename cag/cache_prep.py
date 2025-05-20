import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import disk_offload
from utils.data_loader import load_company_faq
from dotenv import load_dotenv
from transformers.cache_utils import DynamicCache
import time

torch.serialization.add_safe_globals([DynamicCache])

load_dotenv()

MODEL_NAME = os.getenv("HF_MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
FAQ_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'company-faq.md'))

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_tokenizer(model_name, hf_token):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        token=hf_token
        ).cpu()
    # disk_offload(model, offload_dir="offload")  # Removed to avoid meta tensor error
    return tokenizer, model

def preprocess_knowledge(model, tokenizer, prompt):
    """
    Prepare knowledge KV cache for CAG.
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        prompt: The knowledge to preprocess (string)
    Returns:
        DynamicCache: KV Cache
    """
    device = model.device if hasattr(model, 'device') else torch.device('cpu')
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    past_key_values = DynamicCache()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False
        )
    return outputs.past_key_values

def prepare_kvcache(model, tokenizer, documents, filepath, answer_instruction=None):
    # Prepare the knowledges kvcache
    """
    Prepare and save the KV cache for the given documents (FAQ).
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        documents: The FAQ/context string
        filepath: Path to save the cache file
        answer_instruction: Optional instruction for the assistant
    Returns:
        kv_cache: The DynamicCache object
        prep_time: Time taken to prepare the cache (seconds)
    """
    if answer_instruction is None:
        answer_instruction = "Answer the question with a super short answer."
    prompt = f"""
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are an assistant for giving short answers based on given context.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Context information is below.
------------------------------------------------
{documents}
------------------------------------------------
{answer_instruction}
Question:
"""
    # Get the knowledge cache
    t1 = time.time()
    kv_cache = preprocess_knowledge(model, tokenizer, prompt)
    print("kvlenn: ", kv_cache.key_cache[0].shape[-2])
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(kv_cache, filepath)
    t2 = time.time()
    prep_time = t2 - t1
    return kv_cache, prep_time

def generate_answer_with_cache(model, tokenizer, question, kv_cache, answer_instruction=None, max_new_tokens=128):
    """
    Generate an answer to the question using the precomputed FAQ KV cache.
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        question: User question (string)
        kv_cache: DynamicCache object (precomputed FAQ context)
        answer_instruction: Optional instruction for the assistant
        max_new_tokens: Maximum number of tokens to generate
    Returns:
        answer: The generated answer (string)
    """
    if answer_instruction is None:
        answer_instruction = "Answer the question with a super short answer."
    prompt = f"""
{question}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
    device = model.device if hasattr(model, 'device') else torch.device('cpu')
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = generate_with_cache(model, input_ids, kv_cache, max_new_tokens=max_new_tokens)
    answer = tokenizer.decode(output[0], skip_special_tokens=True, temperature=None)
    return answer.strip()

def generate_with_cache(model, input_ids, past_key_values, max_new_tokens=50):
    device = model.device if hasattr(model, 'device') else torch.device('cpu')
    input_ids = input_ids.to(device)
    output_ids = input_ids.clone()
    next_token = input_ids
    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True
        )
        next_token_logits = outputs.logits[:, -1, :]
        next_token = next_token_logits.argmax(dim=-1).unsqueeze(-1)
        next_token = next_token.to(device)
        
        past_key_values = outputs.past_key_values
        
        output_ids = torch.cat([output_ids, next_token], dim=1)
        
        if next_token.item() == model.config.eos_token_id:
            break
    return output_ids[:, input_ids.shape[-1]:]

def clean_up(kv, origin_len):
    for i in range(len(kv.key_cache)):
        kv.key_cache[i] = kv.key_cache[i][:, :, :origin_len, :]
        kv.value_cache[i] = kv.value_cache[i][:, :, :origin_len, :]

if __name__ == "__main__":
    print(f"Loading model: {MODEL_NAME}")
    tokenizer, model = load_model_and_tokenizer(MODEL_NAME, HF_TOKEN)
    device = get_device()
    print(f"Using device: {device}")
    model.to_empty(device=device)

    print(f"Loading FAQ from: {FAQ_PATH}")
    faq_text = load_company_faq(FAQ_PATH)
    print(f"Loaded FAQ length: {len(faq_text)} characters") 

    # Step: Prepare and save the KV cache for the FAQ
    cache_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_cache', 'cache_knowledges.pt'))
    
    print(f"Preparing KV cache and saving to: {cache_path}")
    kv_cache, prep_time = prepare_kvcache(model, tokenizer, faq_text, cache_path)
    print(f"KV cache prepared and saved. Preparation time: {prep_time:.2f} seconds") 

    # Compute the original context length for cache truncation
    origin_len = kv_cache.key_cache[0].shape[-2]

    # Sample questions
    questions = [
        "what is PALO IT?",
        "what technical stacks PALO can offer?"
    ]
    for i, q in enumerate(questions, 1):
        clean_up(kv_cache, origin_len)
        print(f"\nQ{i}: {q}")
        answer = generate_answer_with_cache(model, tokenizer, q, kv_cache)
        print(f"A{i}: {answer}")