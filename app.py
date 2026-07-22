from pathlib import Path
import ast
import zipfile

import nltk
import pandas as pd
import streamlit as st
from nrclex import NRCLex
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(page_title="CinePulse", page_icon="🎥", layout="centered")


@st.cache_resource
def prepare_nltk() -> None:
    """Download only the tokenizer resources NRCLex needs."""
    resources = (
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    )

    for lookup_path, package_name in resources:
        try:
            nltk.data.find(lookup_path)
        except LookupError:
            downloaded = nltk.download(package_name, quiet=True)
            if not downloaded:
                raise RuntimeError(
                    f"NLTK could not download the required resource: {package_name}"
                )


def extract_names(value, limit=None):
    """Safely extract name values from TMDB JSON-like columns."""
    if not isinstance(value, str) or not value.strip():
        return []

    try:
        records = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []

    names = [
        record.get("name", "")
        for record in records
        if isinstance(record, dict) and record.get("name")
    ]
    return names[:limit] if limit else names


def extract_director(value):
    if not isinstance(value, str) or not value.strip():
        return ""

    try:
        crew = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return ""

    for member in crew:
        if isinstance(member, dict) and member.get("job") == "Director":
            return member.get("name", "")
    return ""


def detect_emotion(text):
    if not isinstance(text, str) or not text.strip():
        return "neutral"

    try:
        top_emotions = NRCLex(text).top_emotions
    except Exception:
        return "neutral"

    if not top_emotions or top_emotions[0][1] <= 0:
        return "neutral"
    return top_emotions[0][0]


@st.cache_data(show_spinner="Loading and analysing the movie dataset...")
def load_and_prepare_data():
    """Load both CSV files directly from the ZIP included in the repository."""
    data_zip = Path(__file__).resolve().parent / "data" / "tmdb_5000_credits.zip"

    if not data_zip.exists():
        raise FileNotFoundError(
            "Dataset not found. Expected data/tmdb_5000_credits.zip in the repository."
        )

    with zipfile.ZipFile(data_zip) as archive:
        required_files = {
            "tmdb_5000_movies.csv",
            "tmdb_5000_credits.csv",
        }
        missing_files = required_files.difference(archive.namelist())
        if missing_files:
            raise FileNotFoundError(
                "The dataset ZIP is missing: " + ", ".join(sorted(missing_files))
            )

        with archive.open("tmdb_5000_movies.csv") as movies_file:
            movies_df = pd.read_csv(movies_file)
        with archive.open("tmdb_5000_credits.csv") as credits_file:
            credits_df = pd.read_csv(credits_file)

    movies_df = movies_df.merge(credits_df, on="title", how="inner")
    movies_df["overview"] = movies_df["overview"].fillna("")
    movies_df["genres"] = movies_df["genres"].apply(extract_names)
    movies_df["keywords"] = movies_df["keywords"].apply(extract_names)
    movies_df["cast"] = movies_df["cast"].apply(lambda value: extract_names(value, 3))
    movies_df["director"] = movies_df["crew"].apply(extract_director)

    movies_df["tags"] = movies_df.apply(
        lambda row: " ".join(
            row["genres"]
            + row["keywords"]
            + row["cast"]
            + [row["director"], row["overview"]]
        ),
        axis=1,
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_features=12000)
    tfidf_matrix = vectorizer.fit_transform(movies_df["tags"])
    movies_df["emotion"] = movies_df["overview"].apply(detect_emotion)

    return movies_df, tfidf_matrix


prepare_nltk()

try:
    movies_df, tfidf_matrix = load_and_prepare_data()
except Exception as error:
    st.error("The application could not load its movie data.")
    st.exception(error)
    st.stop()


st.markdown(
    """
    <style>
    .stApp {
        background:
            linear-gradient(rgba(0,0,0,0.68), rgba(0,0,0,0.68)),
            url('https://images.unsplash.com/photo-1601758123927-35b2a43bb9c2?auto=format&fit=crop&w=1950&q=80');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: white;
    }
    h1, h2, h3 { color: #ffdd00; }
    .stButton > button {
        border-radius: 10px;
        min-height: 45px;
        font-size: 16px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


MOOD_GROUPS = {
    "happy": {"joy", "positive", "trust", "anticipation"},
    "sad": {"sadness", "negative"},
    "anger": {"anger", "negative"},
    "fear": {"fear", "negative"},
    "surprise": {"surprise", "anticipation"},
    "relaxed": {"trust", "positive"},
    "neutral": {"neutral"},
    "bored": {"neutral", "sadness"},
    "disgust": {"disgust", "negative"},
    "joy": {"joy", "positive"},
    "heartbroken": {"sadness", "negative"},
    "frustrated": {"anger", "negative"},
}


def recommend_movies_by_mood(title, user_mood, number=5):
    title_matches = movies_df.index[
        movies_df["title"].str.casefold() == title.casefold()
    ].tolist()
    if not title_matches:
        return []

    selected_index = title_matches[0]
    scores = cosine_similarity(
        tfidf_matrix[selected_index], tfidf_matrix
    ).ravel()
    ranked_indices = scores.argsort()[::-1]
    allowed_emotions = MOOD_GROUPS.get(user_mood, {"neutral"})

    preferred = []
    fallback = []

    for movie_index in ranked_indices:
        if movie_index == selected_index:
            continue

        recommendation = {
            "title": movies_df.iloc[movie_index]["title"],
            "score": round(float(scores[movie_index]), 3),
            "mood": movies_df.iloc[movie_index]["emotion"],
            "poster_path": movies_df.iloc[movie_index].get("poster_path"),
        }

        fallback.append(recommendation)
        if recommendation["mood"] in allowed_emotions:
            preferred.append(recommendation)

        if len(preferred) >= number:
            break

    # Avoid an empty screen when the dataset has too few exact mood matches.
    selected = preferred[:]
    used_titles = {item["title"] for item in selected}
    for item in fallback:
        if len(selected) >= number:
            break
        if item["title"] not in used_titles:
            selected.append(item)
            used_titles.add(item["title"])

    return selected[:number]


st.title("🎥 CinePulse")
st.caption("Movie recommendations based on similarity and your current mood")

movie_titles = sorted(movies_df["title"].dropna().unique().tolist())
movie_input = st.text_input("Enter a movie", placeholder="Example: Avatar")

selected_movie = None
if movie_input.strip():
    matching_titles = [
        title for title in movie_titles if movie_input.casefold() in title.casefold()
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
        "How are you feeling right now?", list(mood_options.keys())
    )

    if st.button("Recommend Movies", type="primary"):
        recommendations = recommend_movies_by_mood(
            selected_movie, mood_options[mood_label], 5
        )

        if not recommendations:
            st.warning("No recommendations were found for that movie.")
        else:
            st.subheader(f"🎯 Recommendations for {selected_movie}")
            for recommendation in recommendations:
                poster_path = recommendation["poster_path"]
                if isinstance(poster_path, str) and poster_path.strip():
                    st.image(
                        f"https://image.tmdb.org/t/p/w200{poster_path}",
                        width=120,
                    )
                st.markdown(
                    f"**{recommendation['title']}**  \n"
                    f"Detected emotion: `{recommendation['mood']}` · "
                    f"Similarity: `{recommendation['score']}`"
                )
                st.divider()
