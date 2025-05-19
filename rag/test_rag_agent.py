from rag.db import get_qdrant_client, BIO_COLLECTION
from rag.retrieve import retrieve_biography
from rag.agent import generate_answer

if __name__ == "__main__":
    client = get_qdrant_client()
    test_query = "What are previous companies of Ryu Natthapat?"
    result = retrieve_biography(test_query, client, BIO_COLLECTION)
    if result:
        print(f"Retrieved page: {result['page']}")
        print(f"Score: {result['score']}")
        print(f"Context:\n{result['text'][:500]}...\n")
        answer = generate_answer(test_query, result['text'])
        print(f"\nGenerated Answer:\n{answer}\n")
    else:
        print("No relevant biography found.") 
        
        
        
# PYTHONPATH=. python3 rag/test_rag_agent.py