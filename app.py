import streamlit as st
from file_parser import FileParser
from embedder import Embedder
from ollama_chat import OllamaChat

class RAGSystemApp:
    def __init__(self):
        """Initialize the system components and session state."""
        self.parser = FileParser()
        self.embedder = Embedder('thenlper/gte-large')  # Initialize with the embedding model
        self.chat = OllamaChat()
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize session state variables for chat history and selected models."""
        if "messages" not in st.session_state:
            st.session_state["messages"] = []
        if "model" not in st.session_state:
            st.session_state["model"] = ""

    def display_sidebar(self):
        """Display sidebar options for model selection, file upload, and embedding generation."""
        st.sidebar.title("Customize the RAG System")
        
        # Dropdown for embedding and LLM model selection
        embedding_model = st.sidebar.selectbox("Select the embedding model", ['Phi3.5'])
        llm_model = st.sidebar.selectbox("Select the LLM model", ["Phi3.5"])

        # File uploader widget for PDF and TXT files
        uploaded_file = st.sidebar.file_uploader("Choose a file", type=['pdf', 'txt'])

        # Button to trigger embedding generation
        generate_embeddings = st.sidebar.button("Generate Embeddings")
        
        return uploaded_file, generate_embeddings

    def handle_file_upload(self, uploaded_file):
        """Handle file upload, parse it, and generate embeddings."""
        if uploaded_file:
            # Parse the file using the appropriate parser based on file type
            text = self.parser.parse_file(uploaded_file)

            if text:
                st.write(f"Extracted Text from {uploaded_file.name}:")
                st.text_area("Document Content", value=text, height=200)

                # Generate embeddings from the parsed text
                embeddings = self.embedder.generate_embeddings(text)
                st.write("Generated Embeddings (first 5 values):", embeddings[:5])
            else:
                st.write("No text found in the document or unsupported file format.")
    
    def chat_interface(self):
        """Display and manage the chat interface using Ollama."""
        # Retrieve available models from Ollama and select the model in the sidebar
        models = self.chat.list_models()
        st.session_state["model"] = st.sidebar.selectbox("Choose your model", models)
        
        # Display chat history if any
        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Function to generate model responses using Ollama
        def model_res_generator():
            for chunk in self.chat.generate_response():
                yield chunk

        # Accept user input and generate a response from the selected LLM model
        if prompt := st.chat_input("What is on your mind?"):
            st.session_state["messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                message = st.write_stream(model_res_generator())
                st.session_state["messages"].append({"role": "assistant", "content": message})

    def run(self):
        """Main method to run the Streamlit app."""
        # Set up the page configuration
        st.set_page_config(page_title="RAG System", page_icon="🧠", layout="wide")
        st.title("RAG System Output Flowchart")

        # Sidebar interactions for model selection, file uploading, and embedding generation
        uploaded_file, generate_embeddings = self.display_sidebar()

        # Handle file uploading and embedding generation when the button is clicked
        if generate_embeddings:
            self.handle_file_upload(uploaded_file)

        # Manage the chat interaction interface
        self.chat_interface()

if __name__ == '__main__':
    # Create an instance of the app and run it
    app = RAGSystemApp()
    app.run()
