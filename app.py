import streamlit as st
import pandas as pd
import pickle
import difflib

# Load precomputed models and data
try:
    movies_list = pd.read_pickle('output/movies_list.pkl')
    similarity = pd.read_pickle('output/similarity.pkl')
    algo = pickle.load(open('output/surprise_model.pkl', 'rb'))
    trainset = pd.read_pickle('output/trainset.pkl')
except FileNotFoundError as e:
    st.error(f"Error: {e}. Ensure main.py has been run to generate output files.")
    st.stop()

# Load ratings for user validation
try:
    ratings = pd.read_csv('ratings.csv')
except FileNotFoundError:
    st.error("Error: ratings.csv not found in the current directory.")
    st.stop()

# Hybrid recommendation function
def hybrid_recommend(user_id, movie_title, n=5):
    try:
        close_matches = difflib.get_close_matches(movie_title, movies_list['title'], n=1, cutoff=0.8)
        if not close_matches:
            close = difflib.get_close_matches(movie_title, movies_list['title'], n=3, cutoff=0.6)
            return f"Movie '{movie_title}' not found. Closest matches: {close if close else 'None'}"
        movie_title = close_matches[0]
        movie_index = movies_list[movies_list['title'] == movie_title].index[0]
        
        distance = sorted(list(enumerate(similarity[movie_index])), reverse=True, key=lambda x: x[1])[1:n+10]
        similar_movies = [movies_list.iloc[i[0]] for i in distance]

        if user_id not in trainset._raw2inner_id_users:
            return f"User ID {user_id} not found."
        recommendations = []
        for movie in similar_movies:
            movie_id = movie['movieId']
            if movie_id in trainset._raw2inner_id_items:
                pred = algo.predict(user_id, movie_id)
                recommendations.append((movie['title'], pred.est))

        recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:n]
        return recommendations if recommendations else "No recommendations available."
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit app
st.title('Hybrid Movie Recommender System')

# User input
user_id = st.number_input('Enter User ID:', min_value=1, max_value=int(ratings['userId'].max()), step=1)
selected_movie = st.selectbox('Select a Movie:', options=movies_list['title'].tolist())

# Get recommendations
if st.button('Get Recommendations'):
    recommendations = hybrid_recommend(user_id, selected_movie)
    if isinstance(recommendations, str):
        st.write(recommendations)
    else:
        st.write(f"Recommendations for User {user_id} based on '{selected_movie}':")
        for movie, rating in recommendations:
            st.write(f"{movie} (Predicted Rating: {rating:.2f})")