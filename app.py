import streamlit as st
import os

from config import MODEL_NAME
from prompts import STRICT_PDF_QA_PROMPT
from embeddings import load_index, retrieve_top_k

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
# Load Vector Store
# --------------------------------------------------

# --------------------------------------------------
# Load Vector Store
# --------------------------------------------------

@st.cache_resource
def load_knowledge_base():
    index, chunks = load_index()
    return index, chunks


with st.spinner("Loading knowledge base..."):
    index, chunks = load_knowledge_base()

if index is None or chunks is None:
    st.error("Vector store missing. Please commit precomputed index.")
    st.stop()


st.success("Knowledge Base Ready ✅")###


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