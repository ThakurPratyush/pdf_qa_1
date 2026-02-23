from google import genai
import numpy as np
import faiss
import pickle
import os
from config import GEMINI_API_KEY
from google import genai

client = genai.Client(
    vertexai=True ,
    credentials=credentials,
    project=creds_dict["project_id"],
    location="us-central1",
)
VECTOR_STORE_PATH = "vector_store/faiss.index"
CHUNKS_PATH = "vector_store/chunks.pkl"
    
def get_embedding(text):
    response = client.models.embed_content(
        model="text-embedding-004",
        contents=[text]
    )
    return np.array(response.embeddings[0].values)

def create_and_save_index(chunks):
    embeddings = [get_embedding(chunk) for chunk in chunks]
    embeddings_array = np.array(embeddings).astype("float32")

    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)

    os.makedirs("vector_store", exist_ok=True)
    faiss.write_index(index, VECTOR_STORE_PATH)

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    return index


def load_index():
    if os.path.exists(VECTOR_STORE_PATH) and os.path.exists(CHUNKS_PATH):
        index = faiss.read_index(VECTOR_STORE_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        return index, chunks
    return None, None


def retrieve_top_k(query, chunks, index, k=3):
    query_embedding = get_embedding(query)
    query_embedding = np.array([query_embedding]).astype("float32")

    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]