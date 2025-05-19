from classifier import classify_query

if __name__ == "__main__":
    test_queries = [
        "Who is Ryu?",
        "How to request a leave on Napta?",
        "What's the weather in Paris?"
    ]
    for q in test_queries:
        label = classify_query(q)
        print(f"Query: {q}\n  Classified as: {label}\n") 