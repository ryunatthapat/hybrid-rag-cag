import os
from openai import OpenAI
from tqdm import tqdm
from qdrant_client.models import PointStruct
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-ada-002")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


def get_embedding(text: str, model: str = EMBED_MODEL) -> list:
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding


def embed_biographies(pages: List[Dict], client_qdrant, collection_name: str):
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
        client_qdrant.upsert(collection_name=collection_name, points=[point]) 