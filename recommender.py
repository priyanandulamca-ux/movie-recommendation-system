import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    return pd.read_csv("data/movies.csv")

def build_similarity_matrix(df):
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split('|'))
    genre_matrix = vectorizer.fit_transform(df['genres'])
    similarity_matrix = cosine_similarity(genre_matrix)
    return similarity_matrix

def recommend(movie_title):
    df = load_data()
    similarity_matrix = build_similarity_matrix(df)

    if movie_title not in df['title'].values:
        return []

    index = df[df['title'] == movie_title].index[0]
    similarity_scores = list(enumerate(similarity_matrix[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    for i in similarity_scores[1:4]:
        recommendations.append(df.iloc[i[0]]['title'])

    return recommendations
