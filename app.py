import streamlit as st
from recommender import recommend, load_data

st.set_page_config(page_title="Movie Recommendation System")

st.title("🎬 Movie Recommendation System")
st.write("Content-Based Movie Recommendation using Machine Learning")

df = load_data()
movie_list = df['title'].tolist()

selected_movie = st.selectbox("Select a Movie", movie_list)

if st.button("Recommend"):
    results = recommend(selected_movie)

    if results:
        st.subheader("Recommended Movies:")
        for movie in results:
            st.write("👉", movie)
    else:
        st.write("No recommendations found.")