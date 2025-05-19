import os
import openai
import time
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)
print("OPENAI_API_KEY", OPENAI_API_KEY)
SYSTEM_PROMPT = (
    "You are a classifier agent. Given a user query, classify it as one of the following: "
    "'biography' (if the question is about a person, employees information, experience, skills, or background)."
    "'faq' (if the question is about company information, services, policies, or general facts), "
    "or 'unknown' (if it is unrelated or unclear). "
    "Respond with only one word: biography, faq, or unknown."
)

MODEL = "gpt-4o-mini"


def classify_query(query: str, max_retries: int = 2) -> str:
    """
    Classify the query as 'biography', 'faq', or 'unknown' using OpenAI.
    """
    print("MODEL", MODEL)
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": query}
                ],
                temperature=0.1,
                max_tokens=10,
            )
            label = response.choices[0].message.content.strip().lower()
            if label in {"biography", "faq", "unknown"}:
                return label
            return "unknown"
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            else:
                print(f"[Classifier] OpenAI API error: {e}")
                return "unknown" 