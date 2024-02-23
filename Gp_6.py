#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load KPrototypes model
#with open(r'kprototypes_model.pkl', 'rb') as f:
#    kproto = pickle.load(f)

# Load clustered data
with open(r'finalized_model.pkl', 'rb') as f:
    netflix_data = pickle.load(f)

# Load Netflix data
netflix = pd.read_csv(r"C:\Users\mange\OneDrive\Desktop\GP___6\netflix_titles.csv", encoding='latin1')

# Load TF-IDF Vectorizer and TF-IDF matrix
vectorizer = TfidfVectorizer()
df = pd.read_csv(r"C:\Users\mange\OneDrive\Desktop\GP___6\netflix_titles.csv", encoding='latin1')
df['cleaned_description'] = df['description'].apply(lambda x: x.lower())
tfidf_matrix = vectorizer.fit_transform(df['cleaned_description'])

# Class for Movie Recommendation using TF-IDF
class MovieRecommender:
    def __init__(self):
        self.recommended_movies = []

    def get_recommendation(self, user_input):
        cleaned_user_input = user_input.lower()
        user_tfidf_vector = vectorizer.transform([cleaned_user_input])
        cos_similarities = cosine_similarity(user_tfidf_vector, tfidf_matrix)

        for idx, _ in enumerate(cos_similarities):
            if df.loc[idx, 'title'] in self.recommended_movies:
                cos_similarities[0][idx] = 0

        best_match_index = cos_similarities.argmax()
        best_match_movie_title = df.loc[best_match_index, 'title']
        best_match_description = df.loc[best_match_index, 'description']

        self.recommended_movies.append(best_match_movie_title)

        return best_match_movie_title, best_match_description

# Function to recommend movies based on clusters
def recommend_movies(movie_title, netflix_data, netflix):
    movie_cluster = netflix_data.loc[netflix_data['title'].str.lower() == movie_title.lower(), 'kcluster'].values[0]
    recommended_movies = netflix[netflix['kcluster'] == movie_cluster]
    recommended_movies = recommended_movies[recommended_movies['title'].str.lower() != movie_title.lower()]
    cluster_centroid = netflix_data.loc[netflix_data['kcluster'] == movie_cluster, netflix_data.columns != 'kcluster'].mean()
    recommended_movies['distance'] = recommended_movies.apply(lambda row: np.linalg.norm(row.drop(['title', 'kcluster']) - cluster_centroid), axis=1)
    top_recommendation = recommended_movies.sort_values(by='distance').iloc[0]
    return top_recommendation['title']

# Streamlit UI
def main():
    st.title("Movie Recommendation Web-App")
    st.sidebar.title("Choose Recommendation Type")
    recommendation_type = st.sidebar.radio("Select recommendation type:", ("Content-Based", "Cluster-Based"))

    if recommendation_type == "Content-Based":
        st.write("# Content-Based Movie Recommendation")
        user_input = st.text_input("Enter Movies you watched: ")
        if st.button("Get Recommendation"):
            if user_input:
                recommender = MovieRecommender()
                movie_title, movie_description = recommender.get_recommendation(user_input)
                st.write("Best matching movie title:", movie_title)
                st.write("Description:", movie_description)
            else:
                st.warning("Please enter a query.")

    elif recommendation_type == "Cluster-Based":
        st.write("# Cluster-Based Movie Recommendation")
        movie_title = st.text_input("Enter a movie title:")
        if st.button("Get Recommendation"):
            if movie_title:
                recommendation = recommend_movies(movie_title, netflix_data, netflix)
                st.write(f"Recommended Movie: {recommendation}")
            else:
                st.warning("Please enter a movie title.")

if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




