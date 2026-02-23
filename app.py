import numpy as np
import faiss
import os
import pickle

INDEX_PATH = "vector_store/index.faiss"
CHUNKS_PATH = "vector_store/chunks.pkl"


def get_embedding(text, client):
    response = client.models.embed_content(
        model="text-embedding-004",
        contents=[text],
    )
    return np.array(response.embeddings[0].values)


def create_and_save_index(chunks, client):
    embeddings = [get_embedding(chunk, client) for chunk in chunks]
    dimension = len(embeddings[0])

    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    os.makedirs("vector_store", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    return index


def load_index():
    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        return index, chunks

    return None, None


def retrieve_top_k(query, chunks, index, client, k=3):
    query_embedding = get_embedding(query, client)
    D, I = index.search(np.array([query_embedding]), k)
    return [chunks[i] for i in I[0]]