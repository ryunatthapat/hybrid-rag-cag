import sys
import os
from utils.data_loader import load_biographies
from rag.db import get_qdrant_client, ensure_biographies_collection, BIO_COLLECTION
from rag.embed import embed_biographies

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'biographies.md'))

if __name__ == "__main__":
    print("Loading biographies...")
    pages = load_biographies(DATA_PATH)
    print(f"Loaded {len(pages)} biography pages.")

    print("Connecting to Qdrant...")
    client = get_qdrant_client()
    ensure_biographies_collection(client)

    print("Embedding and indexing biographies...")
    embed_biographies(pages, client, BIO_COLLECTION)
    print("Done embedding and indexing biographies into Qdrant!") 