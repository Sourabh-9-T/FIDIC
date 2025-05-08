
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re

st.title("FIDIC Clause Search")

url = 'https://raw.githubusercontent.com/Sourabh-9-T/FIDIC/refs/heads/main/FIDIC_CSV.csv'

@st.cache_data
def load_data():
    return pd.read_csv(url)

df = load_data()

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

@st.cache_resource
def encode_clauses():
    embeddings = model.encode(df['clause_text'].tolist(), show_progress_bar=True)
    return np.array(embeddings).astype("float32")

clause_embeddings = encode_clauses()
index = faiss.IndexFlatL2(clause_embeddings.shape[1])
index.add(clause_embeddings)

threshold_options = {
    "Only the most relevant clauses": 0.5,
    "Include generally relevant clauses": 1.0,
    "Also include loosely relevant clauses": 1.5,
    "Include everything, even weak matches": 2.0
}

selected_label = st.selectbox("Choose relevance level for results", list(threshold_options.keys()))
distance_threshold = threshold_options[selected_label]

# Helper to get clause by number
def get_clause_by_number(clause_number):
    row = df[df['sub_clause_number_1'] == clause_number]
    if not row.empty:
        return {
            'heading': row.iloc[0]['sub_clause_number_1_heading'],
            'text': row.iloc[0]['clause_text']
        }
    return None

# Helper to find referenced clauses
def extract_references(text):
    matches = re.findall(r"Sub-Clause (\d+(?:\.\d+)*)", text)
    return list(set(matches))

# Search logic
def search_clauses(query, top_k=50, threshold=distance_threshold):
    query_embedding = model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if dist <= threshold:
            results.append({
                'sub_clause_number': df.iloc[idx]['sub_clause_number_1'],
                'heading': df.iloc[idx]['sub_clause_number_1_heading'],
                'text': df.iloc[idx]['clause_text'],
                'distance': round(float(dist), 4)
            })
    return results

# Justified text style helper
def styled_text(text):
    return f"<div style='text-align: justify;'>{text}</div>"

# User input
query = st.text_input("Enter a phrase (e.g. 'delay damages')")

if query:
    with st.spinner("Searching..."):
        results = search_clauses(query, top_k=50, threshold=distance_threshold)

    if results:
        st.success(f"Found {len(results)} result(s):")
        for res in results:
            st.markdown(f"### Clause {res['sub_clause_number']} – {res['heading']}")
            st.markdown(styled_text(res['text']), unsafe_allow_html=True)
            st.caption(f"Similarity distance: {res['distance']}")

            # Extract and display references
            referenced_numbers = extract_references(res['text'])
            if referenced_numbers:
                st.markdown("**🔗 Referenced Clauses (Summarized):**")
                for ref in referenced_numbers:
                    ref_clause = get_clause_by_number(ref)
                    if ref_clause:
                        summary = ref_clause['text'][:200] + "..." if len(ref_clause['text']) > 200 else ref_clause['text']
                        st.markdown(f"- **Clause {ref}** *(summary)*: {summary}")
                        with st.expander(f"View full Clause {ref}"):
                            st.markdown(f"**Clause {ref} – {ref_clause['heading']}**")
                            st.markdown(styled_text(ref_clause['text']), unsafe_allow_html=True)

            st.markdown("---")
    else:
        st.warning("No relevant results found at this relevance level.")
