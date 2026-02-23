import os
import numpy as np
import faiss
import pickle


# --------------------------------------------------
# Absolute Project Root
# --------------------------------------------------

# This gets: /mount/src/pdf_qa/main

# This gets: /mount/src/pdf_qa


INDEX_PATH = os.path.join(os.getcwd(), "vector_store", "faiss.index")
CHUNKS_PATH = os.path.join(os.getcwd(), "vector_store", "chunks.pkl")


# --------------------------------------------------
# Embedding Function
# --------------------------------------------------

def get_embedding(text, client):
    response = client.models.embed_content(
        model="text-embedding-004",
        contents=[text],
    )
    return np.array(response.embeddings[0].values)


# --------------------------------------------------
# Create and Save Index (Only used locally)
# --------------------------------------------------

def create_and_save_index(chunks, client):
    embeddings = [get_embedding(chunk, client) for chunk in chunks]
    dimension = len(embeddings[0])

    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    return index


# --------------------------------------------------
# Load Precomputed Index (Cloud uses this)
# --------------------------------------------------

def load_index():
    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        return index, chunks

    return None, None


# --------------------------------------------------
# Retrieval
# --------------------------------------------------

def retrieve_top_k(query, chunks, index, client, k=3):
    query_embedding = get_embedding(query, client)
    D, I = index.search(np.array([query_embedding]), k)
    return [chunks[i] for i in I[0]]