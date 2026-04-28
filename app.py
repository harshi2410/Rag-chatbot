import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Config
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

# Models
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
generator = pipeline(
    "text-generation",
    model="google/flan-t5-base"
)

st.title("🤖 RAG Chatbot")
st.write("Upload a text file and ask questions from it.")

uploaded_file = st.file_uploader("Upload TXT File", type=["txt"])
query = st.text_input("Ask a question")

if uploaded_file:
    text = uploaded_file.read().decode("utf-8")

    chunks = text.split("\n")
    chunks = [c for c in chunks if c.strip() != ""]

    embeddings = embed_model.encode(chunks)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    st.success("File processed successfully!")

    if query:
        query_embedding = embed_model.encode([query])

        distances, indices = index.search(np.array(query_embedding), 2)

        retrieved = [chunks[i] for i in indices[0]]
        context = " ".join(retrieved)

        prompt = f"""
Context: {context}

Question: {query}

Give a short answer:
"""

        with st.spinner("Thinking..."):
           response = generator(prompt, max_length=100, do_sample=False)

        answer = response[0]["generated_text"].strip() 

        st.write("### 🤖 AI Answer:")
        st.write(answer)

        st.write("### 📄 Retrieved Context:")
        st.write(context)