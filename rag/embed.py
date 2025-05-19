import os
import openai
from tqdm import tqdm
from qdrant_client.models import PointStruct
from typing import List, Dict

EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-ada-002")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY


def get_embedding(text: str, model: str = EMBED_MODEL) -> list:
    resp = openai.Embedding.create(input=[text], model=model)
    return resp["data"][0]["embedding"]


def embed_biographies(pages: List[Dict], client, collection_name: str):
    """
    For each biography page, get embedding and upsert into Qdrant.
    """
    for page in tqdm(pages, desc="Embedding biographies"):
        page_id = page["page"]
        text = page["text"]
        embedding = get_embedding(text)
        point = PointStruct(
            id=page_id,
            vector=embedding,
            payload={"page": page_id, "text": text}
        )
        client.upsert(collection_name=collection_name, points=[point]) 