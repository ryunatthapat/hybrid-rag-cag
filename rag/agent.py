import os
from openai import OpenAI
from dotenv import load_dotenv
import time

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = os.getenv("OPENAI_ANSWER_MODEL", "gpt-4")
client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = (
    "You are an expert assistant. Answer the user's question using ONLY the provided context. "
    "If the answer is not in the context, say you don't know. Be concise and accurate."
)


def generate_answer(query: str, context: str, model: str = DEFAULT_MODEL, max_retries: int = 2) -> str:
    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=256,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            else:
                print(f"[RAG Agent] OpenAI API error: {e}")
                return "[Error: Could not generate answer]" 