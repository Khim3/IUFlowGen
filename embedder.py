from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name):
        """Initialize the embedding model."""
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, text):
        """Generate embeddings for the given text."""
        return self.model.encode(text).tolist()  # Convert to list for easier use
