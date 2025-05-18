import sys
import os
import inspect
import logging
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc

from utils import *

config = load_config()

def get_txt_file_from_dir(directory):
    """Finds the first `.txt` file in the given directory."""
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            return os.path.join(directory, file)
    return None 

def process_text(working_dir):
   # WORKING_DIR = '/home/tttung/Khiem/thesis/working'

    #logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=ollama_model_complete,
        llm_model_name='phi4',
        llm_model_max_async=4,
        llm_model_max_token_size=32768,
        llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 32768}},
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts, embed_model="nomic-embed-text", host="http://localhost:11434"
            ),
        ),
    )
    
    
    document_path = get_txt_file_from_dir(working_dir)

    if document_path:
        with open(document_path, "r", encoding="utf-8") as f:
            rag.insert(f.read())
    

if __name__ == "__main__":
    if len(sys.argv) < 2:  # Only require `working_dir`
        print("Usage: python3 text_processing.py '<working_dir>'")
        sys.exit(1)

    working_dir = sys.argv[1]
    process_text(working_dir)

