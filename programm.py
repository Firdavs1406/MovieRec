import streamlit as st
import requests
import pickle
import pandas as pd
import random
from sklearn.metrics.pairwise import cosine_similarity

with open('cv.pkl', 'rb') as f:
    cv = pickle.load(f)
    
with open('indices.pkl', 'rb') as f:
    indices = pickle.load(f)

with open('movies_list.pkl', 'rb') as f:
    filtered_df = pickle.load(f)

def get_movie_details(movie_id):
    response = requests.get(
        f'https://api.themoviedb.org/3/movie/{movie_id}?api_key=ccaa561c8bbe4cff97a35aae1cbc48ca&language=en-US'
    )
    data = response.json()
    poster_url = "https://image.tmdb.org/t/p/w500/" + data['poster_path']
    overview = data['overview']
    release_date = data['release_date']
    runtime = data['runtime']
    return poster_url, overview, release_date, runtime

def get_recommendations(movie_title):
    idx = filtered_df[filtered_df['title'].str.contains(movie_title, case=False)].index
    if len(idx) == 0:
        st.error("Movie not found. Please, try another title.")
        return []
    
    idx = idx[0]
    similarity_scores = cosine_similarity(cv[idx], cv).flatten()
    similar_movie_indices = similarity_scores.argsort()[-7:-1][::-1]
    return filtered_df['title'].iloc[similar_movie_indices].tolist()

def show_movie_info(title, movie_id):
    poster_url, overview, release_date, runtime = get_movie_details(movie_id)
    rating = filtered_df[filtered_df['id'] == movie_id]['vote_average'].values[0]
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(poster_url, width=170)
    with col2:
        st.write(f"**Title:** {title}")
        st.write(f"**Description:** {overview}")
        st.write(f"**Release Date:** {release_date}")
        st.write(f"**Rating:** {rating:.1f}/10")
        st.write(f"**Runtime:** {runtime} min.")
    st.write("---")

def get_random_popular_movies(n=7):
    top_movies = filtered_df.nlargest(50, 'popularity')
    random_popular = top_movies.sample(n=n)
    return random_popular

st.title("🎬 Movie Recommendations System")

st.subheader("🔍 Find Similar Movies")
col1, col2 = st.columns([3, 1])
with col1:
    movie_title = st.selectbox("Choose movie title:", filtered_df['title'].values)
with col2:
    search_button = st.button("Get Recommendations", type="primary")

if search_button:
    recommendations = get_recommendations(movie_title)
    if recommendations:
        st.subheader("🎯 Movies you might like:")
        for title in recommendations:
            movie_id = filtered_df[filtered_df['title'] == title]['id'].values[0]
            show_movie_info(title, movie_id)
else:
    st.markdown("---")
    st.subheader("📈 Popular Movies Today")
    popular_movies = get_random_popular_movies()
    for _, movie in popular_movies.iterrows():
        show_movie_info(movie['title'], movie['id'])