# app.py

import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Title
st.title("FIDIC Clause Search")

# GitHub CSV link
url = 'https://raw.githubusercontent.com/Sourabh-9-T/FIDIC/refs/heads/main/FIDIC_CSV.csv'

# Load CSV
@st.cache_data
def load_data():
    return pd.read_csv(url)

df = load_data()

# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Encode clause embeddings
@st.cache_resource
def encode_clauses():
    embeddings = model.encode(df['clause_text'].tolist(), show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    return embeddings

clause_embeddings = encode_clauses()

# Build FAISS index
index = faiss.IndexFlatL2(clause_embeddings.shape[1])
index.add(clause_embeddings)

# Search function with sorting option
def search_clauses(query, top_k=20, sort_by="Similarity"):
    query_embedding = model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        results.append({
            'sub_clause_number': df.iloc[idx]['sub_clause_number_1'],
            'heading': df.iloc[idx]['sub_clause_number_1_heading'],
            'text': df.iloc[idx]['clause_text'],
            'distance': dist
        })

    if sort_by == "Clause":
        def clause_key(clause):
            parts = str(clause['sub_clause_number']).split('.')
            return tuple(int(part) if part.isdigit() else 0 for part in parts)
        results.sort(key=clause_key)
    else:
        results.sort(key=lambda x: x['distance'])

    return results

# Input
query = st.text_input("Enter a phrase (e.g. 'delay damages')")

# Sort option
sort_option = st.selectbox(
    "Sort results by:",
    options=["Similarity (Most Relevant First)", "Clause Number (Ascending)"]
)

if query:
    with st.spinner("Searching..."):
        sort_by = "Clause" if "Clause" in sort_option else "Similarity"
        results = search_clauses(query, sort_by=sort_by)

    if results:
        st.success(f"Found {len(results)} result(s):")
        for res in results:
            st.markdown(f"### Clause {res['sub_clause_number']} - {res['heading']}")
            st.write(res['text'])
            st.markdown(f"**Similarity Score (L2 Distance):** {res['distance']:.4f}")
            st.markdown("---")
    else:
        st.warning("No results found.")