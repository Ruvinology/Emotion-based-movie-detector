from __future__ import annotations

import ast
import zipfile
from pathlib import Path
from typing import Any

import nltk
import pandas as pd
import requests
import streamlit as st
from nrclex import NRCLex
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------------------------------------------------------
# Streamlit configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CinePulse",
    page_icon="🎥",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_ZIP = BASE_DIR / "data" / "tmdb_5000_credits.zip"
TMDB_API_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_URL = "https://image.tmdb.org/t/p/w500"
POSTER_PLACEHOLDER = "https://placehold.co/500x750?text=Poster+Unavailable"


# -----------------------------------------------------------------------------
# NLTK setup
# -----------------------------------------------------------------------------
@st.cache_resource
def prepare_nltk() -> None:
    """Download only the tokenizer resources required by NRCLex."""
    resources = (
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    )

    for lookup_path, package_name in resources:
        try:
            nltk.data.find(lookup_path)
        except LookupError:
            if not nltk.download(package_name, quiet=True):
                raise RuntimeError(
                    f"NLTK could not download the required resource: {package_name}"
                )


# -----------------------------------------------------------------------------
# Dataset parsing helpers
# -----------------------------------------------------------------------------
def extract_names(value: Any, limit: int | None = None) -> list[str]:
    """Safely extract name values from TMDB JSON-like CSV columns."""
    if not isinstance(value, str) or not value.strip():
        return []

    try:
        records = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []

    if not isinstance(records, list):
        return []

    names = [
        str(record.get("name", "")).strip()
        for record in records
        if isinstance(record, dict) and record.get("name")
    ]
    return names[:limit] if limit is not None else names


def extract_director(value: Any) -> str:
    """Return the director name from the TMDB crew column."""
    if not isinstance(value, str) or not value.strip():
        return ""

    try:
        crew = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return ""

    if not isinstance(crew, list):
        return ""

    for member in crew:
        if isinstance(member, dict) and member.get("job") == "Director":
            return str(member.get("name", "")).strip()
    return ""


def detect_emotion(text: Any) -> str:
    """Detect the strongest NRC emotion in a movie overview."""
    if not isinstance(text, str) or not text.strip():
        return "neutral"

    try:
        top_emotions = NRCLex(text).top_emotions
    except Exception:
        return "neutral"

    if not top_emotions or top_emotions[0][1] <= 0:
        return "neutral"

    return str(top_emotions[0][0]).lower()


# -----------------------------------------------------------------------------
# Data loading and recommendation model
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and analysing the movie dataset...")
def load_and_prepare_data():
    """Load both TMDB CSV files from the repository ZIP and create TF-IDF data."""
    if not DATA_ZIP.exists():
        raise FileNotFoundError(
            "Dataset not found. Expected data/tmdb_5000_credits.zip."
        )

    with zipfile.ZipFile(DATA_ZIP) as archive:
        required_files = {
            "tmdb_5000_movies.csv",
            "tmdb_5000_credits.csv",
        }
        available_files = set(archive.namelist())
        missing_files = required_files - available_files

        if missing_files:
            raise FileNotFoundError(
                "The dataset ZIP is missing: " + ", ".join(sorted(missing_files))
            )

        with archive.open("tmdb_5000_movies.csv") as movies_file:
            movies = pd.read_csv(movies_file)

        with archive.open("tmdb_5000_credits.csv") as credits_file:
            credits = pd.read_csv(credits_file)

    movies = movies.merge(credits, on="title", how="inner")
    movies = movies.drop_duplicates(subset=["id", "title"]).reset_index(drop=True)

    movies["overview"] = movies["overview"].fillna("").astype(str)
    movies["genres_list"] = movies["genres"].apply(extract_names)
    movies["keywords_list"] = movies["keywords"].apply(extract_names)
    movies["cast_list"] = movies["cast"].apply(
        lambda value: extract_names(value, limit=3)
    )
    movies["director"] = movies["crew"].apply(extract_director)

    # Remove spaces from names so multi-word entities remain connected in TF-IDF.
    movies["tags"] = movies.apply(
        lambda row: " ".join(
            [
                row["overview"],
                *[item.replace(" ", "") for item in row["genres_list"]],
                *[item.replace(" ", "") for item in row["keywords_list"]],
                *[item.replace(" ", "") for item in row["cast_list"]],
                row["director"].replace(" ", ""),
            ]
        ).lower(),
        axis=1,
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=12000,
        ngram_range=(1, 2),
    )
    tfidf_matrix = vectorizer.fit_transform(movies["tags"])

    movies["emotion"] = movies["overview"].apply(detect_emotion)
    movies["release_year"] = pd.to_datetime(
        movies.get("release_date"), errors="coerce"
    ).dt.year

    return movies, tfidf_matrix


MOOD_GROUPS = {
    "happy": {"joy", "positive", "trust", "anticipation"},
    "sad": {"sadness", "negative"},
    "anger": {"anger", "negative"},
    "fear": {"fear", "negative"},
    "surprise": {"surprise", "anticipation"},
    "relaxed": {"trust", "positive"},
    "neutral": {"neutral"},
    "bored": {"neutral", "surprise", "anticipation"},
    "disgust": {"disgust", "negative"},
    "joy": {"joy", "positive", "anticipation"},
    "heartbroken": {"sadness", "negative"},
    "frustrated": {"anger", "negative"},
}


def recommend_movies_by_mood(
    movies: pd.DataFrame,
    matrix,
    title: str,
    user_mood: str,
    number: int = 5,
) -> list[dict[str, Any]]:
    """Recommend similar movies while prioritising the selected mood."""
    title_matches = movies.index[
        movies["title"].str.casefold() == title.casefold()
    ].tolist()

    if not title_matches:
        return []

    selected_index = title_matches[0]
    similarity_scores = cosine_similarity(
        matrix[selected_index], matrix
    ).ravel()
    ranked_indices = similarity_scores.argsort()[::-1]
    allowed_emotions = MOOD_GROUPS.get(user_mood, {"neutral"})

    preferred: list[dict[str, Any]] = []
    fallback: list[dict[str, Any]] = []

    for movie_index in ranked_indices:
        if movie_index == selected_index:
            continue

        row = movies.iloc[movie_index]
        tmdb_id = pd.to_numeric(row.get("id"), errors="coerce")
        release_year = row.get("release_year")

        recommendation = {
            "title": str(row["title"]),
            "tmdb_id": int(tmdb_id) if pd.notna(tmdb_id) else None,
            "score": round(float(similarity_scores[movie_index]), 3),
            "mood": str(row.get("emotion", "neutral")),
            "overview": str(row.get("overview", "")),
            "rating": float(row.get("vote_average", 0) or 0),
            "release_year": int(release_year) if pd.notna(release_year) else None,
            "dataset_poster_path": row.get("poster_path"),
        }

        fallback.append(recommendation)

        if recommendation["mood"] in allowed_emotions:
            preferred.append(recommendation)

        # Keep a larger fallback pool but avoid traversing the whole dataset.
        if len(preferred) >= number and len(fallback) >= number * 4:
            break

    selected = preferred[:number]
    used_titles = {item["title"] for item in selected}

    for item in fallback:
        if len(selected) >= number:
            break
        if item["title"] not in used_titles:
            selected.append(item)
            used_titles.add(item["title"])

    return selected[:number]


# -----------------------------------------------------------------------------
# TMDB poster API
# -----------------------------------------------------------------------------
def read_tmdb_credentials() -> tuple[str | None, str | None]:
    """Read an API Read Access Token or v3 API key from Streamlit secrets."""
    try:
        token = st.secrets.get("TMDB_TOKEN")
        api_key = st.secrets.get("TMDB_API_KEY")
    except Exception:
        return None, None

    token_value = str(token).strip() if token else None
    key_value = str(api_key).strip() if api_key else None
    return token_value, key_value


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_tmdb_poster_path(tmdb_id: int) -> str | None:
    """Retrieve poster_path from TMDB using a movie ID."""
    token, api_key = read_tmdb_credentials()

    if not token and not api_key:
        return None

    headers = {"accept": "application/json"}
    params: dict[str, str] = {}

    if token:
        headers["Authorization"] = f"Bearer {token}"
    elif api_key:
        params["api_key"] = api_key

    try:
        response = requests.get(
            f"{TMDB_API_URL}/movie/{tmdb_id}",
            headers=headers,
            params=params,
            timeout=10,
        )
        response.raise_for_status()
        poster_path = response.json().get("poster_path")
    except (requests.RequestException, ValueError, TypeError):
        return None

    return poster_path if isinstance(poster_path, str) else None


def build_poster_url(
    tmdb_id: int | None,
    dataset_poster_path: Any,
) -> tuple[str, str]:
    """Use the TMDB API first, then fall back to the local dataset poster path."""
    api_poster_path = None

    if tmdb_id is not None:
        api_poster_path = fetch_tmdb_poster_path(tmdb_id)

    if api_poster_path:
        return f"{TMDB_IMAGE_URL}{api_poster_path}", "TMDB API"

    if isinstance(dataset_poster_path, str) and dataset_poster_path.strip():
        return f"{TMDB_IMAGE_URL}{dataset_poster_path.strip()}", "dataset fallback"

    return POSTER_PLACEHOLDER, "placeholder"


# -----------------------------------------------------------------------------
# User interface
# -----------------------------------------------------------------------------
def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                linear-gradient(rgba(5, 8, 20, 0.86), rgba(5, 8, 20, 0.92)),
                url('https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?auto=format&fit=crop&w=1950&q=80');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }
        h1, h2, h3 { color: #ffd54f; }
        [data-testid="stCaptionContainer"] { color: #d7d7d7; }
        .movie-card {
            background: rgba(20, 24, 40, 0.86);
            border: 1px solid rgba(255, 213, 79, 0.25);
            border-radius: 14px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .stButton > button {
            border-radius: 10px;
            min-height: 46px;
            font-size: 16px;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_recommendation_card(recommendation: dict[str, Any]) -> None:
    poster_url, poster_source = build_poster_url(
        recommendation["tmdb_id"],
        recommendation["dataset_poster_path"],
    )

    poster_column, details_column = st.columns([1, 2.4], gap="large")

    with poster_column:
        st.image(poster_url, use_container_width=True)

    with details_column:
        title = recommendation["title"]
        year = recommendation["release_year"]
        heading = f"{title} ({year})" if year else title
        st.subheader(heading)

        metric_one, metric_two, metric_three = st.columns(3)
        metric_one.metric("Similarity", recommendation["score"])
        metric_two.metric("Detected mood", recommendation["mood"].title())
        metric_three.metric("TMDB rating", f"{recommendation['rating']:.1f}/10")

        overview = recommendation["overview"].strip()
        st.write(overview if overview else "No overview is available for this movie.")
        st.caption(f"Poster source: {poster_source}")

    st.divider()


def main() -> None:
    prepare_nltk()
    inject_styles()

    try:
        movies, tfidf_matrix = load_and_prepare_data()
    except Exception as error:
        st.error("The application could not load its movie data.")
        st.exception(error)
        st.stop()

    st.title("🎥 CinePulse")
    st.caption("Movie recommendations based on similarity and your current mood")

    token, api_key = read_tmdb_credentials()
    if not token and not api_key:
        st.info(
            "TMDB credentials are not configured. Posters will use the dataset "
            "fallback. Add TMDB_TOKEN or TMDB_API_KEY in Streamlit Secrets to "
            "retrieve posters through the TMDB API."
        )

    movie_titles = sorted(movies["title"].dropna().astype(str).unique().tolist())
    movie_input = st.text_input(
        "Enter a movie",
        placeholder="Example: Avatar",
    )

    selected_movie: str | None = None

    if movie_input.strip():
        matching_titles = [
            title
            for title in movie_titles
            if movie_input.casefold() in title.casefold()
        ][:100]

        if matching_titles:
            selected_movie = st.selectbox("Select a movie", matching_titles)
        else:
            st.warning("No movie title matches your search.")

    if selected_movie:
        mood_options = {
            "😊 Happy": "happy",
            "😢 Sad": "sad",
            "😡 Angry": "anger",
            "😨 Fearful": "fear",
            "😱 Surprised": "surprise",
            "😌 Relaxed": "relaxed",
            "😐 Neutral": "neutral",
            "😴 Bored": "bored",
            "🤢 Disgusted": "disgust",
            "🤩 Excited": "joy",
            "😭 Heartbroken": "heartbroken",
            "😖 Frustrated": "frustrated",
        }

        mood_label = st.selectbox(
            "How are you feeling right now?",
            list(mood_options.keys()),
        )

        recommendation_count = st.slider(
            "Number of recommendations",
            min_value=3,
            max_value=10,
            value=5,
        )

        if st.button("Recommend Movies", type="primary", use_container_width=True):
            recommendations = recommend_movies_by_mood(
                movies,
                tfidf_matrix,
                selected_movie,
                mood_options[mood_label],
                recommendation_count,
            )

            if not recommendations:
                st.warning("No recommendations were found for that movie.")
            else:
                st.subheader(f"🎯 Recommendations for {selected_movie}")
                for recommendation in recommendations:
                    show_recommendation_card(recommendation)

    st.caption(
        "This product uses the TMDB API but is not endorsed or certified by TMDB."
    )


if __name__ == "__main__":
    main()
