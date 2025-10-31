import streamlit as st
import pandas as pd
import numpy as np
from zipfile import ZipFile
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

@st.cache_data
def load_data():
    zip_path = "ml-100k.zip"
    extract_path = "ml-100k"
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    ratings = pd.read_csv(f"{extract_path}/ml-100k/u.data", sep="\t",
                          names=["userId", "movieId", "rating", "timestamp"])
    movies = pd.read_csv(f"{extract_path}/ml-100k/u.item", sep="|", encoding="latin-1",
                         names=["movieId", "title", "release_date", "video_release_date",
                                "IMDb_URL", "unknown", "Action", "Adventure", "Animation",
                                "Children", "Comedy", "Crime", "Documentary", "Drama",
                                "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
                                "Romance", "Sci-Fi", "Thriller", "War", "Western"])

    movies = movies[["movieId", "title", "Action", "Adventure", "Animation", "Children", "Comedy",
                     "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
                     "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]]

    ratings = ratings.merge(movies, on="movieId")
    genre_cols = movies.columns[2:]
    movies["genres"] = movies[genre_cols].apply(lambda x: " ".join(x.index[x == 1]), axis=1)
    return ratings, movies

ratings, movies = load_data()

@st.cache_data
def build_content_model():
    count = CountVectorizer(stop_words="english")
    count_matrix = count.fit_transform(movies["genres"])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    return cosine_sim

cosine_sim = build_content_model()

def get_content_recommendations(title, cosine_sim=cosine_sim):
    if title not in movies["title"].values:
        return []
    idx = movies[movies["title"] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies["title"].iloc[movie_indices].tolist()

@st.cache_data
def build_collab_model():
    user_movie_matrix = ratings.pivot_table(index="userId", columns="title", values="rating").fillna(0)
    svd = TruncatedSVD(n_components=20, random_state=42)
    latent_matrix = svd.fit_transform(user_movie_matrix.T)
    movie_sim = cosine_similarity(latent_matrix)
    movie_titles = user_movie_matrix.columns
    return movie_sim, movie_titles

movie_sim, movie_titles = build_collab_model()

def get_collab_recommendations(title):
    if title not in movie_titles:
        return []
    idx = list(movie_titles).index(title)
    sim_scores = list(enumerate(movie_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return [movie_titles[i] for i in movie_indices]

def get_hybrid_recommendations(title):
    content_recs = get_content_recommendations(title)
    collab_recs = get_collab_recommendations(title)
    combined = pd.Series(content_recs + collab_recs).value_counts().head(10)
    return combined.index.tolist()

st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")
st.title("🎥 Movie Recommendation System (Hybrid Model)")
st.markdown("Get personalized movie suggestions using **Content-Based + Collaborative Filtering**")

movie_choice = st.selectbox("🎞️ Select a movie:", sorted(movies["title"].unique()))

if st.button("🔍 Get Recommendations"):
    st.subheader("🎯 Content-Based Recommendations")
    st.write(get_content_recommendations(movie_choice))

    st.subheader("🤝 Collaborative Recommendations")
    st.write(get_collab_recommendations(movie_choice))

    st.subheader("🔀 Hybrid Recommendations (Best of Both)")
    st.write(get_hybrid_recommendations(movie_choice))
