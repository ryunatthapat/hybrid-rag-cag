#!/usr/bin/env python3
"""
Main pipeline for the Hybrid RAG-CAG Chatbot.
Handles user queries, classification, routing, and answer generation.
"""

import sys
import time
import os
from typing import Dict, Any

# Import our modules
from classifier.classifier import classify_query, get_openai_client
from answer_cleaner.answer_cleaner import clean_answer

# RAG imports
from rag.db import get_qdrant_client, ensure_biographies_collection, BIO_COLLECTION
from rag.retrieve import retrieve_biography
from rag.agent import generate_answer

# CAG imports
from cag.cache_prep import load_model_and_tokenizer, get_kv_cache, clean_up, generate
from utils.data_loader import load_company_faq

def check_environment():
    """Check if all required environment variables are available."""
    required_vars = {
        'OPENAI_API_KEY': 'OpenAI API for classifier and answer cleaner',
        'HF_TOKEN': 'HuggingFace token for CAG model access'
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"  - {var}: {description}")
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(var)
        print("\nPlease set these environment variables and try again.")
        return False
    
    return True

def initialize_classifier():
    """Initialize the classifier module."""
    print("🔧 Initializing classifier...")
    try:
        client = get_openai_client()
        print("✅ Classifier initialized (OpenAI API client created)")
        return {'client': client}
    except Exception as e:
        print(f"❌ Failed to initialize classifier: {e}")
        return None

def initialize_rag():
    """Initialize the RAG module (vector DB, embeddings, etc.)."""
    print("🔧 Initializing RAG module...")
    
    try:
        # Connect to Qdrant
        print("🔗 Connecting to Qdrant...")
        client = get_qdrant_client()
        
        # Ensure biographies collection exists
        print("📚 Ensuring biographies collection exists...")
        ensure_biographies_collection(client)
        
        print("✅ RAG module initialized")
        return {
            'client': client,
            'collection_name': BIO_COLLECTION
        }
        
    except Exception as e:
        print(f"❌ Failed to initialize RAG module: {e}")
        print("   Make sure Qdrant is running (docker run -p 6333:6333 qdrant/qdrant)")
        return None

def initialize_cag():
    """Initialize the CAG module (load model, build cache, etc.)."""
    print("🔧 Initializing CAG module...")
    
    try:
        # Load FAQ data
        faq_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'raw', 'company-faq.md'))
        print(f"📄 Loading FAQ from: {faq_path}")
        faq_text = load_company_faq(faq_path)
        print(f"✅ FAQ loaded: {len(faq_text)} characters")
        
        # Load model and tokenizer
        model_name = "mistralai/Mistral-7B-Instruct-v0.1"
        hf_token = os.getenv("HF_TOKEN")
        print(f"🤖 Loading model: {model_name}")
        tokenizer, model, device = load_model_and_tokenizer(model_name, hf_token)
        print(f"✅ Model loaded on device: {device}")
        
        # Prepare knowledge prompt for Mistral
        system_prompt = f"""
<|system|>
You are an assistant who provides concise factual answers.
<|user|>
Context:
{faq_text}
Question:
""".strip()
        
        # Build KV cache
        print("🧠 Building KV cache for FAQ...")
        kv_cache, origin_len = get_kv_cache(model, tokenizer, system_prompt)
        print(f"✅ KV cache built. Length: {origin_len}")
        
        return {
            'model': model,
            'tokenizer': tokenizer,
            'device': device,
            'kv_cache': kv_cache,
            'origin_len': origin_len,
            'faq_text': faq_text
        }
        
    except Exception as e:
        print(f"❌ Failed to initialize CAG module: {e}")
        return None

def setup_modules():
    """Initialize all modules required for the pipeline."""
    print("🚀 Setting up Hybrid RAG-CAG Chatbot...")
    print("=" * 50)
    
    # Check environment variables first
    if not check_environment():
        print("❌ Setup failed due to missing environment variables")
        return None
    
    # Initialize modules
    question_classifier = initialize_classifier()
    rag_module = initialize_rag()
    cag_module = initialize_cag()
    
    print("✅ Setup complete!\n")
    
    return {
        'question_classifier': question_classifier,
        'rag_module': rag_module,
        'cag_module': cag_module
    }

def process_query(user_query: str, modules: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a user query through the hybrid RAG-CAG pipeline.
    
    Args:
        user_query: The user's question
        modules: Dictionary containing initialized modules
    
    Returns:
        Dictionary with answer, module used, timing, etc.
    """
    start_time = time.time()
    
    try:
        # Step 1: Classify the query
        print("🔍 Classifying query...")
        
        if modules['question_classifier'] is None:
            print("❌ Classifier not available")
            classification = "unknown"
        else:
            classifier_client = modules['question_classifier']['client']
            classification = classify_query(user_query, client=classifier_client)
        
        print(f"📊 Classification: {classification}")
        
        # Step 2: Route to appropriate module
        if classification == "biography":
            print("🎯 Routing to RAG module...")
            
            if modules['rag_module'] is None:
                raw_answer = "RAG module not available. Please check Qdrant connection."
                module_used = "RAG (Error)"
            else:
                try:
                    # Get RAG components
                    rag = modules['rag_module']
                    client = rag['client']
                    collection_name = rag['collection_name']
                    
                    # Retrieve relevant biography
                    print("📖 Retrieving relevant biography...")
                    result = retrieve_biography(user_query, client, collection_name)
                    
                    if result is None:
                        raw_answer = "No relevant biography found for your query."
                        module_used = "RAG (No Results)"
                    else:
                        # Generate answer using retrieved context
                        context = result['text']
                        biography_page = result['page']
                        print(f"📄 Using biography: {biography_page}")
                        raw_answer = generate_answer(user_query, context)
                        module_used = f"RAG (Page: {biography_page})"
                        
                except Exception as e:
                    raw_answer = f"Error in RAG processing: {str(e)}"
                    module_used = "RAG (Error)"
        
        elif classification == "faq":
            print("🎯 Routing to CAG module...")
            
            if modules['cag_module'] is None:
                raw_answer = "CAG module not available. Please check initialization."
                module_used = "CAG (Error)"
            else:
                try:
                    # Get CAG components
                    cag = modules['cag_module']
                    model = cag['model']
                    tokenizer = cag['tokenizer'] 
                    device = cag['device']
                    kv_cache = cag['kv_cache']
                    origin_len = cag['origin_len']
                    
                    # Clean up cache before use
                    clean_up(kv_cache, origin_len)
                    
                    # Tokenize the question
                    input_ids = tokenizer(user_query + "\n", return_tensors="pt").input_ids.to(device)
                    
                    # Generate answer using the cache
                    output_ids = generate(model, input_ids, kv_cache, max_new_tokens=128)
                    raw_answer = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
                    
                    module_used = "CAG"
                    
                except Exception as e:
                    raw_answer = f"Error in CAG processing: {str(e)}"
                    module_used = "CAG (Error)"
        
        else:  # unknown
            print("❓ Unknown query type")
            raw_answer = "I'm not sure how to answer that. Please ask about biographies or company FAQ."
            module_used = "None"
        
        # Step 3: Clean the answer (if we have a real answer)
        if module_used not in ["None", "RAG (Error)", "CAG (Error)", "RAG (No Results)"]:
            print("✨ Cleaning answer...")
            final_answer = clean_answer(raw_answer, user_query)
        else:
            final_answer = raw_answer
        
        # Calculate timing
        end_time = time.time()
        response_time = end_time - start_time
        
        return {
            'query': user_query,
            'classification': classification,
            'module_used': module_used,
            'raw_answer': raw_answer,
            'final_answer': final_answer,
            'response_time': response_time,
            'success': True
        }
        
    except Exception as e:
        end_time = time.time()
        response_time = end_time - start_time
        
        return {
            'query': user_query,
            'classification': 'error',
            'module_used': 'None',
            'raw_answer': '',
            'final_answer': f"Error processing query: {str(e)}",
            'response_time': response_time,
            'success': False
        }

def main():
    """
    Main CLI loop for the hybrid RAG-CAG chatbot.
    """
    # Setup all modules
    modules = setup_modules()
    
    if modules is None:
        print("❌ Failed to initialize the chatbot. Please fix the issues above and try again.")
        sys.exit(1)
    
    print("🤖 Hybrid RAG-CAG Chatbot")
    print("=" * 40)
    print("Ask questions about biographies or company FAQ.")
    print("Type 'exit' or 'quit' to stop.\n")
    
    while True:
        try:
            # Get user input
            user_query = input("You: ").strip()
            
            # Check for exit conditions
            if user_query.lower() in ['exit', 'quit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_query:
                print("Please enter a question.")
                continue
            
            # Process the query through the pipeline
            result = process_query(user_query, modules)
            
            # Display the result
            print(f"🤖 Answer: {result['final_answer']}")
            print(f"📋 Module: {result['module_used']} | ⏱️  Time: {result['response_time']:.2f}s")
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

if __name__ == "__main__":
    main() 