import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load TF-IDF Vectorizer and TF-IDF matrix
vectorizer = TfidfVectorizer()
df = pd.read_csv(r"C:\Users\mange\OneDrive\Desktop\GP___6\netflix_titles.csv", encoding='latin1')
df['cleaned_description'] = df['description'].apply(lambda x: x.lower())
tfidf_matrix = vectorizer.fit_transform(df['cleaned_description'])

# Function to recommend movies based on user input
def recommend_movies(user_input):
    cleaned_user_input = user_input.lower()
    user_tfidf_vector = vectorizer.transform([cleaned_user_input])
    cos_similarities = cosine_similarity(user_tfidf_vector, tfidf_matrix)

    best_match_indices = np.argsort(cos_similarities)[:, -10:][0][::-1]  # Get indices of top 10 matches
    recommended_movies = [(df.loc[idx, 'title'], df.loc[idx, 'description']) for idx in best_match_indices]

    return recommended_movies

# Streamlit UI
def main():
    st.markdown(
        """
        <style>
        .main {
            background-image: url("https://wallpaperbat.com/img/56305-netflix-picture.jpg");
            background-size: cover;
            padding: 20px;
            height: 100vh;
            color: white;
            font-family: Arial, sans-serif;
        }

        .title {
            text-align: center;
            font-size: 36px;
            margin-bottom: 30px;
        }

        .subtitle {
            text-align: center;
            font-size: 24px;
            margin-bottom: 20px;
        }

        .input-box {
            width: 50%;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
            border: none;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

        .recommendation-button {
            background-color: #FF0000;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            border: none;
            cursor: pointer;
            transition: background-color 0.3s;
        }

        .recommendation-button:hover {
            background-color: #CC0000;
        }

        </style>
        """,
        unsafe_allow_html=True
    )

    st.write('<div class="main">', unsafe_allow_html=True)
    st.write('<div class="title">Netflix Movie Recommendation Web-App</div>', unsafe_allow_html=True)
    st.write('<div class="subtitle">Content-Based Movie Recommendation</div>', unsafe_allow_html=True)

    user_input = st.text_input("Enter Movies you watched:", "")
    if st.button("Get Recommendation"):
        if user_input:
            recommended_movies = recommend_movies(user_input)
            data = {'Movie Title': [movie[0] for movie in recommended_movies], 'Description': [movie[1] for movie in recommended_movies]}
            recommendations_df = pd.DataFrame(data)
            recommendations_df.index += 1  # Start index numbering from 1
            st.table(recommendations_df.reset_index().rename(columns={'index': 'Recommendation Number'}).set_index('Recommendation Number'))
        else:
            st.warning("Please enter a valid entry.")

    st.write('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
