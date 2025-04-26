from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import pypdfium2
from langchain_ollama.embeddings import OllamaEmbeddings
from backend import *
from graphdb import *
from langchain_chroma import Chroma
import chromadb
import uuid
import streamlit as st
config = load_config()

def get_pdf_texts(pdfs_bytes_list):
    return [extract_text_from_pdf(pdf_bytes.getvalue()) for pdf_bytes in pdfs_bytes_list]

def extract_text_from_pdf(pdf_bytes):
    pdf_file = pypdfium2.PdfDocument(pdf_bytes)
    return "\n".join(pdf_file.get_page(page_number).get_textpage().get_text_range() for page_number in range(len(pdf_file)))

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=config["pdf_text_splitter"]["chunk_size"], 
                                              chunk_overlap=config["pdf_text_splitter"]["overlap"],
                                              separators=config["pdf_text_splitter"]["separators"])
    return splitter.split_text(text)

def get_document_chunks(text_list, source=None, add_uuid=False):
    """Convert text into LangChain Document objects, ensuring metadata is always a dictionary."""
    documents = []
    for text in text_list:
        for chunk in get_text_chunks(text):
            metadata = {}  # Always start with an empty dictionary
            
            if source:
                metadata["source"] = source
            if add_uuid:
                metadata["chunk_id"] = str(uuid.uuid4())

            # Ensure metadata is always a dictionary (even if empty)
            documents.append(Document(page_content=chunk, metadata=metadata))
    
    return documents


def create_embeddings(text):
    ollama = OllamaEmbeddings(model=config["ollama"]["embedding_model"])
    return ollama.embed_documents(text)


def load_vectordb(embeddings=OllamaEmbeddings(model=config["ollama"]["embedding_model"])):
    persistent_client = chromadb.PersistentClient(config["chromadb"]["chromadb_path"])

    langchain_chroma = Chroma(
        client=persistent_client,
        collection_name=config["chromadb"]["collection_name"],
        embedding_function=embeddings,
    )

    return langchain_chroma

def retrieve_documents_from_vectordb():
    vector_db = load_vectordb()
    return vector_db.get()['documents']
    

def add_documents_to_db(pdfs_bytes):
    texts = get_pdf_texts(pdfs_bytes)
    documents = get_document_chunks(texts)
    df = convert_document_to_dataframe(documents)
    st.write(df)
    convert_dataframe_to_graph(df)
    vector_db = load_vectordb()
    #vector_db.add_documents(documents)
    st.write("Documents added to VectorDB")