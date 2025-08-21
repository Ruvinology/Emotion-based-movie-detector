import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nrclex import NRCLex
import streamlit as st
import requests

# === Load and Preprocess Data ===
@st.cache_data
def load_and_prepare_data():
    movies_url = f"https://drive.google.com/uc?export=download&id=1Z0yMfy5AiWy6Dx_qUA403LM8EwKDT1Iw"
    credits_url = f"https://drive.google.com/uc?export=download&id=1aQyDihszwK3ucaOEpcVZOyaTCxUs_PVw"

    movies_df = pd.read_csv(movies_url)
    credits_df = pd.read_csv(credits_url)
    movies_df = movies_df.merge(credits_df, left_on='title', right_on='title')

    def extract_names(x):
        try:
            return [d['name'] for d in ast.literal_eval(x)]
        except:
            return []

    movies_df['genres'] = movies_df['genres'].apply(extract_names)
    movies_df['keywords'] = movies_df['keywords'].apply(extract_names)
    movies_df['cast'] = movies_df['cast'].apply(lambda x: [d['name'] for d in ast.literal_eval(x)][:3])

    def get_director(crew_str):
        try:
            crew = ast.literal_eval(crew_str)
            for member in crew:
                if member['job'] == 'Director':
                    return member['name']
            return ''
        except:
            return ''

    movies_df['director'] = movies_df['crew'].apply(get_director)

    def combine_features(row):
        return ' '.join(row['genres']) + ' ' + ' '.join(row['keywords']) + ' ' + ' '.join(row['cast']) + ' ' + row[
            'director'] + ' ' + str(row['overview'])

    movies_df['tags'] = movies_df.apply(combine_features, axis=1)
    movies_df['tags'] = movies_df['tags'].fillna('')

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['tags'])

    similarity = cosine_similarity(tfidf_matrix)

    def detect_emotion(text):
        if not isinstance(text, str) or len(text.strip()) == 0:
            return "neutral"
        emotions = NRCLex(text).top_emotions
        if emotions:
            return emotions[0][0]
        return "neutral"

    movies_df['emotion'] = movies_df['overview'].apply(detect_emotion)

    return movies_df, similarity

# Load data
movies_df, similarity = load_and_prepare_data()

# === Streamlit CSS Styling with Cinematic Background Image ===
st.markdown(
"""
<style>
/* Full-page background image with gradient overlay */
body {
    background: 
        linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)),
        url('https://images.unsplash.com/photo-1601758123927-35b2a43bb9c2?auto=format&fit=crop&w=1950&q=80');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: #ffffff;
}

/* Main container styling */
.css-18e3th9 {
    background: rgba(0, 0, 0, 0.6);  /* semi-transparent dark overlay */
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 8px 16px rgba(0,0,0,0.3);
}

/* Headings */
h1, h2, h3, h4, h5 {
    color: #ffdd00;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #ff416c, #ff4b2b);
    color: white !important;           /* Text always visible */
    border-radius: 10px;
    height: 45px;
    width: 220px;
    font-size: 16px;
    font-weight: bold;
    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    transition: transform 0.2s, color 0.2s;
}
.stButton>button:hover {
    transform: scale(1.05);
    cursor: pointer;
    color: white !important;
}
.stButton>button:active {
    transform: scale(0.98);
    color: white !important;
    background: linear-gradient(90deg, #ff416c, #ff4b2b);
}

/* Text input */
.stTextInput>div>div>input {
    background-color: rgba(240,242,246,255);
    color: #000000;
    border: 1px solid #f0f2f6;
    border-radius: 8px;
    padding: 8px;
    outline: none !important; 
    box-shadow: none !important;
}
.stTextInput>div>div>input:focus {
    outline: none !important;
    box-shadow: none !important;
    border: 1px solid ;
}

/* Selectbox */
.stSelectbox>div>div>div>div {
    background-color: rgba(240,242,246,255);
    color: #000000;
    border: 1px solid #f0f2f6;
    border-radius: 8px;
    outline: none !important;
    box-shadow: none !important;
}
.stSelectbox>div>div>div>div:focus {
    outline: none !important;
    box-shadow: none !important;
    border: 1px solid ;
}

/* Warnings */
.stWarning {
    background-color: rgba(255, 0, 0, 0.3);
    border-left: 5px solid #ff0000;
}
</style>
""",
unsafe_allow_html=True
)



# === Helper Functions ===
def recommend_movies_by_mood(title, user_mood, num_recommendations=5):
    try:
        idx = movies_df[movies_df['title'].str.lower() == title.lower()].index[0]
    except IndexError:
        return []

    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Relaxed mood matching
    mood_groups = {
        "happy": ["happy", "joy", "excited", "trust", "relaxed"],
        "sad": ["sad", "heartbroken", "neutral"],
        "anger": ["anger", "frustrated"],
        "fear": ["fear", "surprised"],
        "bored": ["bored", "neutral"],
        "disgust": ["disgust"],
        "neutral": ["neutral"]
    }

    mood_filtered = [
        (i, score) for i, score in sim_scores[1:]
        if movies_df.iloc[i]['emotion'] in mood_groups.get(user_mood.lower(), [])
    ]

    mood_filtered = mood_filtered[:num_recommendations]

    results = []
    for i, score in mood_filtered:
        results.append({
            'title': movies_df.iloc[i]['title'],
            'score': round(score, 3),
            'mood': movies_df.iloc[i]['emotion'],
            'poster_path': movies_df.iloc[i].get('poster_path', None)  # Use poster if available
        })

    return results

# === Streamlit UI ===
st.title("🎥 CinePulse")

# Auto-complete search bar for movie titles
movie_titles = movies_df['title'].tolist()
movie_input = st.text_input("Enter a movie:", "")

# Filter movie titles based on user input
matching_titles = [title for title in movie_titles if movie_input.lower() in title.lower()]

if movie_input:
    selected_movie = st.selectbox("Select a movie", matching_titles)
else:
    selected_movie = None

if selected_movie:
    st.write(f"Selected movie: **{selected_movie}**")

    # Mood input with emojis
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
        "😶 Speechless": "neutral"
    }

    mood_input_label = st.selectbox("How are you feeling right now?", options=list(mood_options.keys()))
    detected_mood = mood_options[mood_input_label]

    if st.button("Recommend Movies"):
        if not movie_input:
            st.warning("Please enter a movie.")
        else:
            recommendations = recommend_movies_by_mood(selected_movie, detected_mood, 5)

            if recommendations:
                st.subheader(f"🎯 Recommendations for '{selected_movie}':")
                for rec in recommendations:
                    if rec['poster_path']:
                        st.image(f"https://image.tmdb.org/t/p/w200{rec['poster_path']}", width=100)
                    st.markdown(f"**→ {rec['title']}**  | Mood: _{rec['mood']}_ | Score: `{rec['score']}`")
            else:
                st.warning(f"No matching movies found for '{selected_movie}' with mood '{detected_mood}'.")

            if st.button("🔄 Search Again"):
                st.experimental_rerun()
