# Duplicate Question Retrieval (Quora, BEIR)

Compares TF-IDF, BM25, Sentence Transformer (all-MiniLM-L6-v2) and Hybrid RRF for finding duplicate Quora questions.

1. Install: `pip install -r requirements.txt`
2. Run the notebook: open `ir_project.ipynb` and run all cells (Jupyter or Google Colab). This downloads the data, builds the models and saves `results.csv`, `pr_curve.png` and the model files.
3. Start the search app (after step 2): `streamlit run app.py`

The first run takes a few minutes (data download and embeddings). After that, the saved files load instantly.
