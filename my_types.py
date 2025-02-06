from enum import Enum

class CHAIN_TYPE(Enum):
    """
    Enumeration to identify the type of chain the user is working on
    """
    CHAIN_SIMPLE = 1
    CHAIN_RAG = 2

# ollama LLM model used: update here if you want to use a different LLM from Ollama
OLLAMA_MODEL_NAME= "llama3.2"

# ollama EMBEDDINGS model used: update here if you want to use a different EMBEDDINGS from Ollama
OLLAMA_EMBEDDINGS_NAME= "nomic-embed-text"

RAG_URLS = []