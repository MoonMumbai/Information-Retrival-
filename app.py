"""Streamlit search interface for duplicate question retrieval.
Run the notebook (ir_project.ipynb) first: it creates the files loaded below.
Start with:  streamlit run app.py
"""
import re
import pickle
import string
from collections import defaultdict

import numpy as np
import pandas as pd
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

nltk.download("stopwords", quiet=True)

# Same preprocessing as in the notebook
KEEP_WORDS = {"what", "why", "how", "when", "where", "who", "which", "whom", "not", "no"}
STOP_WORDS = set(stopwords.words("english")) - KEEP_WORDS
stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    tokens = [t for t in text.split() if t not in STOP_WORDS]
    return [stemmer.stem(t) for t in tokens]


@st.cache_resource
def load_everything():
    # Loaded only once, then cached by Streamlit
    subset = pickle.load(open("quora_subset.pkl", "rb"))
    tfidf_vectorizer, tfidf_matrix = pickle.load(open("tfidf.pkl", "rb"))
    bm25 = pickle.load(open("bm25.pkl", "rb"))
    corpus_emb = np.load("corpus_emb.npy")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return subset, tfidf_vectorizer, tfidf_matrix, bm25, corpus_emb, model

subset, tfidf_vectorizer, tfidf_matrix, bm25, corpus_emb, model = load_everything()
doc_texts = subset["doc_texts"]


def top_k(scores, k=100):
    best = np.argsort(-scores)[:k]
    return [(i, float(scores[i])) for i in best]

def search_tfidf(query):
    query_vec = tfidf_vectorizer.transform([" ".join(preprocess(query))])
    return top_k(cosine_similarity(query_vec, tfidf_matrix).ravel())

def search_bm25(query):
    return top_k(bm25.get_scores(preprocess(query)))

def search_dense(query):
    return top_k(corpus_emb @ model.encode(query, normalize_embeddings=True))

def rrf_fuse(result_lists, k=5, c=60):
    fused = defaultdict(float)
    for results in result_lists:
        for rank, (doc, _) in enumerate(results, start=1):
            fused[doc] += 1 / (c + rank)
    return sorted(fused.items(), key=lambda x: -x[1])[:k]


st.title("Duplicate Question Retrieval (Quora)")
query = st.text_input("Type a question", "How can I learn Python quickly?")

if st.button("Search") and query.strip():
    tfidf_res, bm25_res, dense_res = search_tfidf(query), search_bm25(query), search_dense(query)
    all_results = {"TF-IDF": tfidf_res[:5], "BM25": bm25_res[:5], "Sentence Transformer": dense_res[:5],
                   "Hybrid RRF": rrf_fuse([tfidf_res, bm25_res, dense_res])}
    # One column per model
    for col, (name, results) in zip(st.columns(4), all_results.items()):
        col.subheader(name)
        for doc, score in results:
            col.markdown(f"**{score:.3f}** · {doc_texts[doc]}")

st.header("Evaluation results (500 test queries)")
st.dataframe(pd.read_csv("results.csv"), hide_index=True)
st.image("pr_curve.png", caption="11-point interpolated Precision-Recall curve")
