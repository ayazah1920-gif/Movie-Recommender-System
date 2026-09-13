"""
Movie Recommender System - Streamlit Frontend
================================================
Content-based movie recommender jo TMDB dataset par based hai.
Deploy-ready for Streamlit Community Cloud.
"""

import os
import pickle

import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Page config (sabse pehle call hona chahiye)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
PLACEHOLDER_POSTER = "https://placehold.co/500x750?text=Poster+Not+Found"

# ---------------------------------------------------------------------------
# Custom CSS — smooth, pyari, thori playful frontend
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(160deg, #1b1024 0%, #2d1b45 45%, #4a2a6d 100%);
        }
        h1, h2, h3, p, span, label, .stMarkdown {
            color: #f5f0ff !important;
        }
            .hero-title {
            font-size: 3.4rem;
            font-weight: 900;
            letter-spacing: 1px;
            background: linear-gradient(90deg, #ff6ec7, #ffb86c, #8ec5ff, #b18cff, #ff6ec7);
            background-size: 300% 300%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0;
            animation: shimmer 6s ease-in-out infinite;
            text-shadow: 0 0 30px rgba(177, 140, 255, 0.35);
        }
        @keyframes shimmer {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        }
        .hero-subtitle {
            color: #cbb8ff !important;
            font-size: 1.05rem;
            margin-top: 0.2rem;
        }
        div[data-testid="stSelectbox"] label {
            font-size: 1.05rem;
            font-weight: 600;
        }
        .movie-card {
            background: rgba(255, 255, 255, 0.06);
            border-radius: 18px;
            padding: 12px;
            text-align: center;
            transition: transform 0.25s ease, box-shadow 0.25s ease;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }
        .movie-card:hover {
            transform: translateY(-8px) scale(1.02);
            box-shadow: 0 12px 24px rgba(177, 140, 255, 0.35);
            border: 1px solid rgba(177, 140, 255, 0.6);
        }
        .movie-card img {
            border-radius: 12px;
            margin-bottom: 10px;
            width: 100%;
        }
        .movie-title {
            font-size: 0.95rem;
            font-weight: 600;
            color: #ffffff !important;
        }
        div.stButton > button {
            background: linear-gradient(90deg, #b18cff, #ff9ecb);
            color: #1b1024;
            font-weight: 700;
            border-radius: 12px;
            border: none;
            padding: 0.6rem 1.4rem;
            transition: transform 0.15s ease;
        }
        div.stButton > button:hover {
            transform: scale(1.03);
            color: #1b1024;
        }
        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Data loading (cached so it only runs once)
# ---------------------------------------------------------------------------
@st.cache_data
def load_artifacts():
    movies_path = os.path.join(ARTIFACTS_DIR, "movie_list.pkl")
    similarity_path = os.path.join(ARTIFACTS_DIR, "similarity.pkl")

    if not (os.path.exists(movies_path) and os.path.exists(similarity_path)):
        return None, None

    with open(movies_path, "rb") as f:
        movies = pickle.load(f)
    with open(similarity_path, "rb") as f:
        similarity = pickle.load(f)

    return movies, similarity


@st.cache_data(show_spinner=False)
def fetch_poster(movie_id: int) -> str:
    """TMDB API se poster URL laata hai. API key na ho to placeholder deta hai."""
    api_key = st.secrets.get("TMDB_API_KEY", os.environ.get("TMDB_API_KEY", ""))
    if not api_key:
        return PLACEHOLDER_POSTER

    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        poster_path = data.get("poster_path")
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except requests.exceptions.RequestException:
        pass
    return PLACEHOLDER_POSTER


def recommend(movie_title: str, movies: pd.DataFrame, similarity):
    """
    `similarity` yahan har movie ke top-20 sab se milte-julte movies ke
    indices ki array hai (preprocess.py dekho) — poori NxN matrix nahi.
    """
    index = movies[movies["title"] == movie_title].index[0]
    top_movie_indices = similarity[index][:5]

    names, posters = [], []
    for i in top_movie_indices:
        movie_id = movies.iloc[i].movie_id
        names.append(movies.iloc[i].title)
        posters.append(fetch_poster(movie_id))
    return names, posters


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.markdown('<p class="hero-title">🎬 MOVIE RECOMMENDER SYSTEM</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">Apni pasandeeda movie choose karo, hum tumhe milti-julti '
    "5 movies suggest karenge ✨</p>",
    unsafe_allow_html=True,
)
st.write("")

movies, similarity = load_artifacts()

with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        "Ye ek **content-based movie recommender system** hai jo movie ke "
        "overview, genres, keywords, cast aur director ke basis par similarity "
        "nikalta hai (Cosine Similarity)."
    )
    st.write("Made with ❤️ using Streamlit + scikit-learn")
    st.divider()
    st.caption(
        "Posters TMDB API se load hote hain. Agar API key set nahi hai to "
        "placeholder image dikhegi."
    )

if movies is None or similarity is None:
    st.warning(
        "⚠️ **artifacts/movie_list.pkl** ya **artifacts/similarity.pkl** nahi milay.\n\n"
        "Pehle `preprocessing/preprocess.py` run karo taake ye files ban sakein.\n"
        "Detail README.md mein hai."
    )
else:
    selected_movie = st.selectbox(
        "🔎 Ek movie select karo:",
        movies["title"].values,
    )

    if st.button("Recommend 🎥"):
        with st.spinner("Best matches dhoondi ja rahi hain..."):
            names, posters = recommend(selected_movie, movies, similarity)

        st.write("")
        cols = st.columns(5)
        for col, name, poster in zip(cols, names, posters):
            with col:
                st.markdown(
                    f"""
                    <div class="movie-card">
                        <img src="{poster}" alt="{name}">
                        <div class="movie-title">{name}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
