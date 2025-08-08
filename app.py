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


# === Helper Functions ===
def detect_user_mood(text):
    emotions = NRCLex(text).top_emotions
    return emotions[0][0] if emotions else "neutral"


def recommend_movies_by_mood(title, user_mood, num_recommendations=5):
    try:
        idx = movies_df[movies_df['title'].str.lower() == title.lower()].index[0]
    except IndexError:
        return []

    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    mood_filtered = [
        (i, score) for i, score in sim_scores[1:]
        if movies_df.iloc[i]['emotion'] == user_mood.lower()
    ]

    mood_filtered = mood_filtered[:num_recommendations]

    results = []
    for i, score in mood_filtered:
        results.append({
            'title': movies_df.iloc[i]['title'],
            'score': round(score, 3),
            'mood': movies_df.iloc[i]['emotion']
        })

    return results


# === Streamlit UI ===
st.title("🎥 Mood-Based Movie Detector")

# Auto-complete search bar for movie titles
movie_titles = movies_df['title'].tolist()  # List of all movie titles in the dataset
movie_input = st.text_input("Enter a movie title:", "")

# Filter movie titles based on user input
matching_titles = [title for title in movie_titles if movie_input.lower() in title.lower()]

# Show the selectbox for the user to choose from the matching titles
if movie_input:
    selected_movie = st.selectbox("Select a movie", matching_titles)
else:
    selected_movie = None

# Display the selected movie
if selected_movie:
    st.write(f"Selected movie: **{selected_movie}**")

    # Detect user mood based on text input
    mood_input = st.text_area("How are you feeling right now? (e.g., I'm tired and stressed)")

    if st.button("Recommend Movies"):
        if not movie_input or not mood_input:
            st.warning("Please enter both movie and mood.")
        else:
            detected_mood = detect_user_mood(mood_input)
            st.write(f"🧠 Detected mood: **{detected_mood}**")

            # Recommend movies based on mood and selected movie
            recommendations = recommend_movies_by_mood(selected_movie, detected_mood, 5)

            if recommendations:
                st.subheader(f"🎯 Top Recommendations for '{selected_movie}' with mood '{detected_mood}':")
                for rec in recommendations:
                    st.markdown(f"**→ {rec['title']}**  | Mood: _{rec['mood']}_ | Score: `{rec['score']}`")
            else:
                st.error(f"No matching movies found for '{selected_movie}' with mood '{detected_mood}'.")



