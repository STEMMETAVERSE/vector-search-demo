app.py
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

docs = [
    "AI is changing education",
    "Machine learning uses data",
    "Robotics combines AI and hardware"
]

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

st.title("🔍 Vector Search Demo")

query = st.text_input("Search")

if st.button("Find"):

    doc_embeddings = model.encode(docs)

    query_embedding = model.encode([query])

    scores = cosine_similarity(
        query_embedding,
        doc_embeddings
    )[0]

    best = docs[scores.argmax()]

    st.success(best)
