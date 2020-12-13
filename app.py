import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import math


@st.cache
def load_data(allow_output_mutation=True):
    users = pd.read_csv("data/users.dat", sep="\t", header=None)
    users.columns = ["user", "gender", "age", "occupation", "zip"]
    ratings = pd.read_csv("data/ratings.dat", sep="\t", header=None)
    ratings.columns = ["user", "movie", "rating", "tc"]
    cross = np.load("data/cross.npz")["cross"]
    movies = pd.read_csv("data/movies.dat", sep="\t", header=None)
    movies.columns = ["movie", "title", "genres"]
    genres_col = movies["genres"].values
    genres = []
    for g in genres_col:
        genres += g.split("|")
    genres = list(set(genres))
    for genre in genres:
        movies[genre] = False
    for i, row in movies.iterrows():
        for genre in row["genres"].split("|"):
            movies.loc[i, genre] = True
    movies.drop("genres", axis="columns", inplace=True)
    return users, movies, ratings, cross


st.sidebar.title("Movie Recommender")

U, M, R, C = load_data()

rated = {
    2858: 0,
    260: 0,
    1196: 0,
    1210: 0,
    480: 0,
    2028: 0,
    589: 0,
    2571: 0,
    1270: 0,
    593: 0,
    1580: 0,
    1198: 0,
    608: 0,
    2762: 0,
    110: 0,
    2396: 0,
    1197: 0,
    527: 0,
    1617: 0,
    1265: 0,
    1097: 0,
    2628: 0,
    2997: 0,
    318: 0,
    858: 0,
}

type_model = st.sidebar.radio(
    "Select the recommendation type", ["Best rated movies", "Item-based CF"]
)

if type_model == "Best rated movies":
    genre = st.sidebar.selectbox("Select a genre", sorted(M.columns[2:]))
    m_ids = M.loc[M[genre] == True, "movie"].values
    sel_ratings = R.loc[R["movie"].isin(m_ids)]
    sel_ratings = sel_ratings.groupby("movie")["rating"].agg(["mean", "count"])
    sel_ratings = sel_ratings.reset_index().sort_values("mean", ascending=False)
    sel_ratings = sel_ratings.loc[sel_ratings["count"] >= 10]
    sel_ratings = sel_ratings.head(50)
    res = M.merge(sel_ratings, on="movie", how="right")[
        ["movie", "mean", "count", "title"]
    ]
    for i, row in res.iterrows():
        st.header(row["title"])
        try:
            st.image("MovieImages/{}.jpg".format(row["movie"]))
        except:
            st.write("No Poster available")
else:
    st.sidebar.subheader("Rate some of the Top-25 movies to get new recommendations")
    for k in rated.keys():
        rated[k] = st.sidebar.slider(
            M.loc[M["movie"] == k, "title"].values[0], 0, 5, 0, 1
        )
    vec = np.zeros((C.shape[1]))
    for k, v in rated.items():
        vec[k] = v
    sim = np.zeros((C.shape[0]))
    for i in range(len(C)):
        pr = stats.pearsonr(vec, C[i])[0]
        if math.isnan(pr):
            pr = 0
        sim[i] = pr
    idx = sim.argsort()[::-1]
    movs = C[idx[:50]].mean(axis=0)
    idx = movs.argsort()[::-1][:50]
    for i in idx:
        st.header(M.loc[M["movie"] == i, "title"].values[0])
        st.image("MovieImages/{}.jpg".format(i))
