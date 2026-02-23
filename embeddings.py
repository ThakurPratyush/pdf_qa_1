import os
import numpy as np
import faiss
import pickle
import hashlib


# --------------------------------------------------
# Paths
# --------------------------------------------------

INDEX_PATH = os.path.join(os.getcwd(), "vector_store", "faiss.index")
CHUNKS_PATH = os.path.join(os.getcwd(), "vector_store", "chunks.pkl")
FINGERPRINT_PATH = os.path.join(os.getcwd(), "vector_store", "fingerprint.txt")


# --------------------------------------------------
# PDF Fingerprint (Detect changes)
# --------------------------------------------------

def compute_pdf_fingerprint(pdf_folder):
    hash_md5 = hashlib.md5()

    for file_name in sorted(os.listdir(pdf_folder)):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(pdf_folder, file_name)
            with open(file_path, "rb") as f:
                hash_md5.update(f.read())

    return hash_md5.hexdigest()


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
# Create and Save Index
# --------------------------------------------------

def create_and_save_index(chunks, client, pdf_folder):
    embeddings = [get_embedding(chunk, client) for chunk in chunks]
    dimension = len(embeddings[0])

    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    os.makedirs("vector_store", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    # Save fingerprint
    fingerprint = compute_pdf_fingerprint(pdf_folder)
    with open(FINGERPRINT_PATH, "w") as f:
        f.write(fingerprint)

    return index


# --------------------------------------------------
# Load Index (Auto change detection)
# --------------------------------------------------

def load_index(pdf_folder):
    if (
        not os.path.exists(INDEX_PATH)
        or not os.path.exists(CHUNKS_PATH)
        or not os.path.exists(FINGERPRINT_PATH)
    ):
        return None, None

    current_fingerprint = compute_pdf_fingerprint(pdf_folder)

    with open(FINGERPRINT_PATH, "r") as f:
        saved_fingerprint = f.read().strip()

    if current_fingerprint != saved_fingerprint:
        print("PDFs changed. Rebuilding index...")
        return None, None

    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    return index, chunks


# --------------------------------------------------
# Retrieval
# --------------------------------------------------

def retrieve_top_k(query, chunks, index, client, k=3):
    query_embedding = get_embedding(query, client)
    D, I = index.search(np.array([query_embedding]), k)
    return [chunks[i] for i in I[0]]