# Hybrid Movie Recommender System

A hybrid movie recommender system combining content-based (TF-IDF on genres) and collaborative filtering (KNNBasic from scikit-surprise). Built with Python, Streamlit, and MovieLens dataset.

## Setup
1. Install dependencies:
   ```bash
   pip install streamlit pandas scikit-surprise numpy==1.26.4

Run main.ipynb to train models and generate output/ files.
Run the Streamlit app:

streamlit run app.py
Files
main.ipynb: Model training and evaluation.
app.py: Streamlit web interface.
movies.csv, ratings.csv: MovieLens dataset.
output/: Model files (excluded from Git due to size).
Performance
RMSE: 0.9847
Precision@5: 0.6610
undefined
Save the file.