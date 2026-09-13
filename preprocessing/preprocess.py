"""
Data Preprocessing Script - Movie Recommender System
======================================================
Ye script TMDB 5000 dataset (movies + credits) ko process karta hai aur
2 pickle files generate karta hai jo Streamlit app use karega:

    artifacts/movie_list.pkl   -> processed movies dataframe
    artifacts/similarity.pkl  -> cosine similarity matrix

USAGE:
    1. Kaggle se dataset download karo:
       https://www.kaggle.com/tmdb/tmdb-movie-metadata
       (tmdb_5000_movies.csv aur tmdb_5000_credits.csv)
    2. Dono CSV files ko `data/` folder mein rakho.
    3. Terminal mein run karo:
       python preprocessing/preprocess.py
"""

import ast
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from nltk.stem.porter import PorterStemmer

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print(
        "[warning] nltk install nahi mila — stemming skip ho jayegi. "
        "Behtar results ke liye `pip install nltk` chala kar dobara run karo."
    )

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

MOVIES_CSV = os.path.join(DATA_DIR, "tmdb_5000_movies.csv")
CREDITS_CSV = os.path.join(DATA_DIR, "tmdb_5000_credits.csv")


def load_data() -> pd.DataFrame:
    if not (os.path.exists(MOVIES_CSV) and os.path.exists(CREDITS_CSV)):
        raise FileNotFoundError(
            "Dataset files nahi milay!\n"
            f"Please '{MOVIES_CSV}' aur '{CREDITS_CSV}' download kar ke data/ folder mein daalo.\n"
            "Dataset link: https://www.kaggle.com/tmdb/tmdb-movie-metadata"
        )

    movies = pd.read_csv(MOVIES_CSV)
    credits = pd.read_csv(CREDITS_CSV)

    # Merge on title
    movies = movies.merge(credits, on="title")

    # Sirf wo columns rakho jo tags banane ke liye chahiye
    movies = movies[
        ["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]
    ]
    movies.dropna(inplace=True)
    return movies


def convert(obj: str) -> list:
    """JSON-like string list se 'name' fields nikalta hai. e.g. genres/keywords."""
    return [item["name"] for item in ast.literal_eval(obj)]


def convert_cast(obj: str, top_n: int = 3) -> list:
    """Cast list se top N actor names nikalta hai."""
    names = []
    for i, item in enumerate(ast.literal_eval(obj)):
        if i >= top_n:
            break
        names.append(item["name"])
    return names


def fetch_director(obj: str) -> list:
    """Crew list se sirf director ka naam nikalta hai."""
    for item in ast.literal_eval(obj):
        if item.get("job") == "Director":
            return [item["name"]]
    return []


def remove_spaces(items: list) -> list:
    """Multi-word names ko ek token banata hai. e.g. 'Science Fiction' -> 'ScienceFiction'"""
    return [i.replace(" ", "") for i in items]


ps = PorterStemmer() if NLTK_AVAILABLE else None


def stem(text: str) -> str:
    if ps is None:
        return text
    return " ".join(ps.stem(word) for word in text.split())


def build_dataset() -> pd.DataFrame:
    movies = load_data()

    movies["genres"] = movies["genres"].apply(convert).apply(remove_spaces)
    movies["keywords"] = movies["keywords"].apply(convert).apply(remove_spaces)
    movies["cast"] = movies["cast"].apply(convert_cast).apply(remove_spaces)
    movies["crew"] = movies["crew"].apply(fetch_director).apply(remove_spaces)
    movies["overview"] = movies["overview"].apply(lambda x: x.split())

    movies["tags"] = (
        movies["overview"]
        + movies["genres"]
        + movies["keywords"]
        + movies["cast"]
        + movies["crew"]
    )

    new_df = movies[["movie_id", "title", "tags"]].copy()
    new_df["tags"] = new_df["tags"].apply(lambda x: " ".join(x).lower())
    new_df["tags"] = new_df["tags"].apply(stem)

    return new_df


TOP_K = 20  # har movie ke liye kitni "similar movies" store karni hain


def build_top_similar(new_df: pd.DataFrame, top_k: int = TOP_K) -> np.ndarray:
    """
    Poori NxN similarity matrix store karne ke bajaye (jo bahut bari file
    banati hai — 4800 movies ke liye ~180MB+), sirf har movie ke top_k
    sab se milte-julte movies ke INDICES store karte hain.
    Isse file size 100x+ chhoti ho jati hai aur app ki recommend()
    speed bhi behtar hoti hai.
    """
    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(new_df["tags"]).toarray()
    similarity = cosine_similarity(vectors)

    n = similarity.shape[0]
    top_indices = np.zeros((n, top_k), dtype=np.int32)

    for i in range(n):
        # apne aap (i) ko exclude kar ke top_k sab se zyada similar movies
        row = similarity[i]
        order = np.argsort(-row)  # descending order
        order = order[order != i][:top_k]
        top_indices[i, : len(order)] = order

    return top_indices


def main():
    print("Dataset load ho raha hai...")
    new_df = build_dataset()

    print("Vectorization + top-similar movies ban rahe hain...")
    top_indices = build_top_similar(new_df)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    with open(os.path.join(ARTIFACTS_DIR, "movie_list.pkl"), "wb") as f:
        pickle.dump(new_df.reset_index(drop=True), f)

    with open(os.path.join(ARTIFACTS_DIR, "similarity.pkl"), "wb") as f:
        pickle.dump(top_indices, f)

    print(f"Done! {len(new_df)} movies process ho gayi hain.")
    print(f"Files bann gayi: {ARTIFACTS_DIR}/movie_list.pkl aur similarity.pkl")


if __name__ == "__main__":
    main()
