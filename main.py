import streamlit as st
import pandas as pd
import numpy as np
import time, ollama

def main():
    st.title("RAG System with Chatbot Interface")
    # create sidebar
    st.sidebar.title("Customize the RAG system")
    # create dropdown for selecting the embedding models
    embedding_model = st.sidebar.selectbox("Select the embedding model", [ "llama3.1", 'Phi3.5'])
    # create a dropdown to select LLM model 
    llm_model = st.sidebar.selectbox("Select the LLM model", ["llama3.1", "Phi3.5"])
    # create a button to upload the data
    uploaded_file = st.sidebar.file_uploader("Choose a file")    
    
    # button to generate embeddings
    generate_embeddings = st.sidebar.button("Generate embeddings")   
    if generate_embeddings:
        st.sidebar.write("Embeddings generated")
    
    # create button
    button = st.sidebar.button("Run")
    if button and embedding_model != "Choose 1 model": 
        st.sidebar.write('Choose model:', embedding_model)
        
   # initialize history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    # init models
    if "model" not in st.session_state:
        st.session_state["model"] = ""

    models = [model["name"] for model in ollama.list()["models"]]
    st.session_state["model"] = st.selectbox("Choose your model", models)
        
    def model_res_generator():
        stream = ollama.chat(
            model=st.session_state["model"],
            messages=st.session_state["messages"],
            stream=True,
        )
        for chunk in stream:
            yield chunk["message"]["content"]

    # Display chat messages from history on app rerun
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        # add latest message to history in format {role, content}
        st.session_state["messages"].append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message = st.write_stream(model_res_generator())
            st.session_state["messages"].append({"role": "assistant", "content": message})
    
if __name__ == '__main__':
    main()