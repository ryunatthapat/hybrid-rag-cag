from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
BIO_COLLECTION = "biographies"
DEFAULT_VECTOR_SIZE = 1536  # OpenAI ada-002 embedding size


def get_qdrant_client():
    """
    Connect to the local Qdrant instance.
    """
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def ensure_biographies_collection(client, collection_name=BIO_COLLECTION, vector_size=DEFAULT_VECTOR_SIZE):
    """
    Ensure the biographies collection exists in Qdrant.
    """
    if collection_name not in [c.name for c in client.get_collections().collections]:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        ) 