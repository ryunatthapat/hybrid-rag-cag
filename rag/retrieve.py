import os
from openai import OpenAI
from qdrant_client.models import SearchRequest
from typing import Dict
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-ada-002")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client_openai = OpenAI(api_key=OPENAI_API_KEY)


def get_query_embedding(query: str, model: str = EMBED_MODEL) -> list:
    response = client_openai.embeddings.create(input=[query], model=model)
    return response.data[0].embedding


def retrieve_biography(query: str, client_qdrant, collection_name: str) -> Dict:
    embedding = get_query_embedding(query)
    search_result = client_qdrant.search(
        collection_name=collection_name,
        query_vector=embedding,
        limit=10,
        score_threshold=0.5
    )
    if not search_result:
        return None
    point = search_result[0]
    return {
        "page": point.payload.get("page"),
        "text": point.payload.get("text"),
        "score": point.score
    } 
    
# PYTHONPATH=. python3 rag/test_retrieve.py