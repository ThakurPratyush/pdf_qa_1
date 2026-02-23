import streamlit as st
import os
from config import GEMINI_API_KEY, MODEL_NAME
from pdf_utils import extract_text_from_pdf, chunk_text
from prompts import STRICT_PDF_QA_PROMPT
from embeddings import create_and_save_index, load_index, retrieve_top_k

import json
from google.oauth2 import service_account
from google import genai
import json
from google.oauth2 import service_account
from google import genai
import streamlit as st

# creds_dict = json.loads(st.secrets["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
creds_dict = json.loads(st.secrets["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
credentials = service_account.Credentials.from_service_account_info(
    creds_dict,
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)

client = genai.Client(
    credentials=credentials,
    project=creds_dict["pdf-rag-project-488220"],
    location="us-central1"
)
PDF_FOLDER = "data/knowledge_base"

st.set_page_config(page_title="PDF Knowledge Chatbot", layout="wide")
st.title("📚 Knowledge Base Chatbot (Strict Mode)")

# ----------------------------
# Load & Index PDFs (Persistent + Cached)
# ----------------------------
@st.cache_resource
def load_knowledge_base():
    # Try loading saved index
    index, chunks = load_index()

    if index is not None and chunks is not None:
        return chunks, index

    # Otherwise create new index
    all_text = ""

    for file_name in os.listdir(PDF_FOLDER):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(PDF_FOLDER, file_name)
            with open(file_path, "rb") as f:
                all_text += extract_text_from_pdf(f)

    chunks = chunk_text(all_text)
    index = create_and_save_index(chunks)

    return chunks, index


with st.spinner("Loading knowledge base..."):
    chunks, index = load_knowledge_base()

st.success("Knowledge Base Ready ✅")

# ----------------------------
# Chat History
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ----------------------------
# Chat Input
# ----------------------------
user_input = st.chat_input("Ask a question about the documents")

if user_input:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Retrieve relevant chunks
    relevant_chunks = retrieve_top_k(user_input, chunks, index)
    context = "\n\n".join(relevant_chunks)

    # Create strict prompt
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

            # Show source excerpts
            with st.expander("📎 Source Excerpts"):
                for i, chunk in enumerate(relevant_chunks):
                    st.markdown(f"**Source {i+1}:**")
                    st.write(chunk[:800] + "...")

    # Save assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )