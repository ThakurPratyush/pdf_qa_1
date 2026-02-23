import streamlit as st
import os

from config import MODEL_NAME
from pdf_utils import extract_text_from_pdf, chunk_text
from prompts import STRICT_PDF_QA_PROMPT
from embeddings import create_and_save_index, load_index, retrieve_top_k

from google.oauth2 import service_account
from google import genai


# --------------------------------------------------
# Streamlit Page Setup
# --------------------------------------------------

st.set_page_config(page_title="PDF Knowledge Chatbot", layout="wide")
st.title("📚 Knowledge Base Chatbot (Strict Mode)")
st.write("App initialized successfully ✅")


# --------------------------------------------------
# Vertex AI Setup
# --------------------------------------------------

if "google_credentials" not in st.secrets:
    st.error("Google credentials not found in Streamlit secrets.")
    st.stop()

creds_dict = dict(st.secrets["google_credentials"])

credentials = service_account.Credentials.from_service_account_info(
    creds_dict,
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)

client = genai.Client(
    vertexai=True,
    credentials=credentials,
    project=creds_dict["project_id"],
    location="us-central1",
)


# --------------------------------------------------
# PDF Folder Check
# --------------------------------------------------

PDF_FOLDER = "data/knowledge_base"

if not os.path.exists(PDF_FOLDER):
    st.error(f"PDF folder not found: {PDF_FOLDER}")
    st.stop()

pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]

if not pdf_files:
    st.error("No PDF files found in knowledge base folder.")
    st.stop()

st.write(f"Found {len(pdf_files)} PDF files.")


# --------------------------------------------------
# Load Knowledge Base
# --------------------------------------------------

@st.cache_resource
def load_knowledge_base():
    index, =  load_index()

    if index is not None and chunks is not None:
        return chunks, index

    st.error("Vector store missing.")
    st.stop()


# --------------------------------------------------
# Chat History
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# Chat Input
# --------------------------------------------------

user_input = st.chat_input("Ask a question about the documents")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # Retrieve relevant chunks
    relevant_chunks = retrieve_top_k(
        user_input,
        chunks,
        index,
        client
    )

    context = "\n\n".join(relevant_chunks)

    final_prompt = STRICT_PDF_QA_PROMPT.format(
        context=context,
        question=user_input
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=final_prompt,
            )

            answer = response.text
            st.markdown(answer)

            with st.expander("📎 Source Excerpts"):
                for i, chunk in enumerate(relevant_chunks):
                    st.markdown(f"**Source {i+1}:**")
                    st.write(chunk[:800] + "...")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )