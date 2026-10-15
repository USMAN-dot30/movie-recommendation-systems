import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬"
)


# Title
st.title("🎬 Movie Recommendation System")
st.write("Find movies similar to your favorite movie!")


# Load movie dataset
movies = pd.read_csv("movies.csv")


# Combine genres and description
movies["content"] = (
    movies["genres"].fillna("") + " " +
    movies["description"].fillna("")
)


# Convert text into numbers
tfidf = TfidfVectorizer(stop_words="english")

tfidf_matrix = tfidf.fit_transform(
    movies["content"]
)


# Calculate similarity
similarity = cosine_similarity(tfidf_matrix)