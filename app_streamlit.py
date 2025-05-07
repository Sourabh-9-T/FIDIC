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

# Search function with distance values
def search_clauses(query, top_k=20):
    query_embedding = model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for dist, i in zip(distances[0], indices[0]):
        results.append({
            'sub_clause_number': df.iloc[i]['sub_clause_number_1'],
            'heading': df.iloc[i]['sub_clause_number_1_heading'],
            'text': df.iloc[i]['clause_text'],
            'distance': float(dist)
        })
    return results

# Input box
query = st.text_input("Enter a phrase (e.g. 'delay damages')")

if query:
    with st.spinner("Searching..."):
        results = search_clauses(query)

    if results:
        st.success(f"Found {len(results)} result(s):")
        for res in results:
            st.markdown(f"### Clause {res['sub_clause_number']} - {res['heading']}")
            st.markdown(f"`Similarity (L2 distance): {res['distance']:.4f}`")
            st.write(res['text'])
            st.markdown("---")
    else:
        st.warning("No results found.")