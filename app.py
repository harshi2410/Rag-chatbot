import re

import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline


st.set_page_config(page_title="Chatbot", page_icon="🤖")

INTENT_RULES = (
    (
        "greeting",
        re.compile(r"^(?:hi|hello|hey|good\s*(?:morning|afternoon|evening))\b", re.I),
        "Hello! Ask me about book genres, or upload a text file and ask me questions about it.",
    ),
    (
        "help",
        re.compile(r"\b(?:help|what can you do|how do you work|capabilities)\b", re.I),
        "I can chat about book genres and answer questions using an uploaded .txt document. "
        "Upload a file, then ask a question about its contents.",
    ),
    (
        "book_genres",
        re.compile(r"\b(?:book|books|genre|genres|reading|recommend(?:ation|ations)?)\b", re.I),
        "Popular genres include romance, fantasy, science fiction, mystery, thriller, "
        "historical fiction, horror, and action-adventure. Nonfiction options include "
        "biography, memoir, self-help, and true crime. Tell me what themes you enjoy "
        "and I can help narrow it down.",
    ),
    (
        "goodbye",
        re.compile(r"^(?:bye|goodbye|see you|thanks|thank you)\b", re.I),
        "You're welcome! Come back whenever you have another question.",
    ),
)


@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_answer_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")


@st.cache_data(show_spinner="Preparing your document...")
def embed_document(text):
    chunks = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]
    if not chunks:
        return [], np.empty((0, 0), dtype=np.float32)

    embeddings = load_embed_model().encode(chunks, normalize_embeddings=True)
    return chunks, np.asarray(embeddings, dtype=np.float32)


def match_intent(message):
    normalized_message = message.strip()
    for intent, pattern, response in INTENT_RULES:
        if pattern.search(normalized_message):
            return intent, response
    return None, None


def answer_document_question(question, chunks, embeddings):
    query_embedding = np.asarray(
        load_embed_model().encode([question], normalize_embeddings=True),
        dtype=np.float32,
    )
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    result_count = min(3, len(chunks))
    similarities, indices = index.search(query_embedding, result_count)

    if similarities[0][0] < 0.2:
        return (
            "I couldn't find information about that in the uploaded document. "
            "Try asking about a detail that appears in the file."
        ), ""

    context = "\n".join(chunks[index] for index in indices[0] if index >= 0)
    prompt = (
        "Answer the question using only the context below. If the context does not "
        "contain the answer, say: I couldn't find that in the uploaded document.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    result = load_answer_model()(prompt, max_new_tokens=100, do_sample=False)
    answer = result[0]["generated_text"].strip()
    if not answer:
        answer = "I couldn't create an answer from the retrieved document text."
    return answer, context


st.title("🤖 Conversational RAG Chatbot")
st.write(
    "Chat about book genres, ask for help, or upload a text document to ask questions "
    "about its contents."
)

uploaded_file = st.file_uploader("Upload a TXT file", type=["txt"])
document_chunks = []
document_embeddings = np.empty((0, 0), dtype=np.float32)

if uploaded_file:
    try:
        document_text = uploaded_file.getvalue().decode("utf-8-sig")
    except UnicodeDecodeError:
        st.error("The uploaded file must be a UTF-8 encoded text file.")
    else:
        if not document_text.strip():
            st.warning("The uploaded file is empty. Add some text before asking questions.")
        else:
            document_chunks, document_embeddings = embed_document(document_text)
            st.success(f"Document ready: {len(document_chunks)} text segments indexed.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("context"):
            with st.expander("Retrieved document text"):
                st.write(message["context"])

user_message = st.chat_input("Type your message")
if user_message:
    st.session_state.messages.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.markdown(user_message)

    _, rule_response = match_intent(user_message)
    retrieved_context = ""

    if rule_response:
        response = rule_response
    elif document_chunks:
        with st.spinner("Searching the document..."):
            response, retrieved_context = answer_document_question(
                user_message,
                document_chunks,
                document_embeddings,
            )
    else:
        response = (
            "I’m not sure how to answer that yet. I can help with greetings, book genres, "
            "or questions about an uploaded text file. Try asking for help or upload a .txt file."
        )

    with st.chat_message("assistant"):
        st.markdown(response)
        if retrieved_context:
            with st.expander("Retrieved document text"):
                st.write(retrieved_context)

    st.session_state.messages.append(
        {"role": "assistant", "content": response, "context": retrieved_context}
    )
