import ollama
import streamlit as st
class OllamaChat:
    def list_models(self):
        """List available models from Ollama."""
        return [model["name"] for model in ollama.list()["models"]]

    def generate_response(self):
        """Generate a chat response from Ollama."""
        stream = ollama.chat(
            model=st.session_state["model"],
            messages=st.session_state["messages"],
            stream=True,
        )
        for chunk in stream:
            yield chunk["message"]["content"]
