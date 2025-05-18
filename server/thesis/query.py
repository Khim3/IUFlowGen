import sys, re
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

def markdown_to_plain_text(md):
    # Remove headings (e.g., ###, ##, #)
    md = re.sub(r'^#{1,6}\s*', '', md, flags=re.MULTILINE)

    # Remove bold and italic (**bold**, *italic*, _italic_, __bold__)
    md = re.sub(r'(\*\*|__)(.*?)\1', r'\2', md)
    md = re.sub(r'(\*|_)(.*?)\1', r'\2', md)

    # Remove ordered list numbers (e.g., 1., 2.)
    md = re.sub(r'^\s*\d+\.\s*', '', md, flags=re.MULTILINE)

    # Remove bullet points (e.g., -, *, +)
    md = re.sub(r'^\s*[-*+]\s+', '', md, flags=re.MULTILINE)

    # Remove inline code (`code`)
    md = re.sub(r'`(.+?)`', r'\1', md)

    # Remove blockquotes
    md = re.sub(r'^>\s+', '', md, flags=re.MULTILINE)

    # Strip extra newlines
    md = re.sub(r'\n{2,}', '\n\n', md)

    return md.strip()

def query_graph_vector(query_text, working_dir):
 
    if not os.path.exists(working_dir):
        os.makedirs(working_dir, exist_ok=True)

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
    # find all txt file in working dir
    
    document_path = get_txt_file_from_dir(working_dir)

    if document_path:
        with open(document_path, "r", encoding="utf-8") as f:
            rag.insert(f.read())
            
    query_response = rag.query(query_text, param=QueryParam(mode="hybrid"))
    query_response = markdown_to_plain_text(query_response)
    print(query_response)

    return query_response

if __name__ == "__main__":
    if len(sys.argv) < 3:  # Expecting query_text and working_dir
        sys.exit(1)
    query_text = sys.argv[1]
    working_dir = sys.argv[2]
    query_graph_vector(query_text, working_dir)

