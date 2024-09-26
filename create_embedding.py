from sentence_transformers import SentenceTransformer

model = SentenceTransformer('thenlper/gte-large')
def get_embedding(text: str) -> list[float]:
    if text is None or not text.strip():
    #    print('No text found')
        return []
    embedding = model.encode(text)
    return embedding.tolist()

