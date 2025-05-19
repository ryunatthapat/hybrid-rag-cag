# Hybrid RAG-CAG Chatbot

This project implements a modular, extensible chatbot system that supports both Retrieval-Augmented Generation (RAG) and Cache-Augmented Generation (CAG) for answering queries about company biographies and FAQs.

## Features
- **RAG**: Uses Chroma vector database and OpenAI embeddings to retrieve and answer questions about expert biographies.
- **CAG**: Uses HuggingFace, torch, and DynamicCache to preload and answer company FAQ queries efficiently.
- **Automatic Question Classification**: Routes queries to the appropriate module using a lightweight LLM-based classifier.
- **CLI Interface**: Interact with the chatbot via the command line, with colored output and response timing.
- **Logging**: All queries, responses, timings, and retrievals are logged for later analysis.

## Project Structure
```
hybrid-rag-cag/
├── cli/
├── classifier/
├── rag/
├── cag/
├── data/
├── logs/
├── utils/
├── requirements.txt
├── Dockerfile
├── README.md
└── main.py
```

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set your OpenAI API key as an environment variable:
   ```bash
   export OPENAI_API_KEY=your-key-here
   ```
3. Place your data files (`biographies.md`, `company-faq.md`) in the `data/` directory.

## Core Dependencies
- torch
- transformers
- chromadb
- openai
- tqdm
- colorama
- python-dotenv

## Notes
- The system is designed to run on CPU by default and is compatible with Mac (Apple Silicon/Intel).
- All configuration (such as API keys) should be set via environment variables.

See the implementation plan in `../plan.md` for more details on architecture and next steps. 