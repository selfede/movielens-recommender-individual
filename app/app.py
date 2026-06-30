import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="MovieLens Recommender", page_icon="🎬", layout="wide")

@st.cache_data
def load_data():
    import io, zipfile, urllib.request
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    with urllib.request.urlopen(url) as r:
        zf = zipfile.ZipFile(io.BytesIO(r.read()))
    ratings = pd.read_csv(zf.open("ml-latest-small/ratings.csv"))
    movies  = pd.read_csv(zf.open("ml-latest-small/movies.csv"))
    tags    = pd.read_csv(zf.open("ml-latest-small/tags.csv"))
    return ratings, movies, tags

@st.cache_data
def preprocess(ratings, movies, tags):
    df = ratings.merge(movies, on="movieId")

    #combining tags per movie into single string -> usable as text feature for tfidf
    tag_str = (tags.groupby("movieId")["tag"]
                   .apply(lambda x: " ".join(x.astype(str)))
                   .reset_index()
                   .rename(columns={"tag": "tags"}))

    movies_ext = movies.merge(tag_str, on="movieId", how="left")
    movies_ext["tags"] = movies_ext["tags"].fillna("")
    #replacing | with space so tfidf treats each genre as separate token
    movies_ext["genres_clean"] = movies_ext["genres"].str.replace("|", " ", regex=False)
    movies_ext["content"] = movies_ext["genres_clean"] + " " + movies_ext["tags"]

    return df, movies_ext

@st.cache_data
def build_tfidf(movies_ext):
    #500 features is enough to capture genre + tag signal without too much noise
    tfidf = TfidfVectorizer(max_features=500)
    tfidf_matrix = tfidf.fit_transform(movies_ext["content"])
    return tfidf_matrix

@st.cache_data
def build_user_item(df):
    #pivot -> rows=users, cols=movies, values=ratings. fillna(0) = no rating
    user_item = df.pivot_table(index="userId", columns="movieId", values="rating").fillna(0)
    return user_item

def bayesian_avg(ratings_series, C=50, m=None):
    #bayesian avg pulls rating toward global mean when vote count is low
    #C=50 means a movie needs ~50 ratings before its avg is trusted on its own
    if m is None:
        m = ratings_series.mean()
    v = len(ratings_series)
    R = ratings_series.mean()
    return (v * R + C * m) / (v + C)

def non_personalized_recs(df, movies, method="popularity", n=10, genre=None):
    stats = df.groupby("movieId").agg(count=("rating","count"), mean=("rating","mean")).reset_index()
    result = stats.merge(movies[["movieId","title","genres"]], on="movieId")

    #genre filter applied before ranking
    if genre and genre != "All":
        result = result[result["genres"].str.contains(genre, na=False)]

    if method == "popularity":
        top = result.nlargest(n, "count")
    elif method == "bayesian":
        global_mean = df["rating"].mean()
        result["bayes"] = result.apply(
            lambda r: bayesian_avg(df[df["movieId"]==r["movieId"]]["rating"], m=global_mean), axis=1)
        top = result.nlargest(n, "bayes")
    else:
        #min 50 ratings before we trust the avg score
        top = result[result["count"] >= 50].nlargest(n, "mean")

    return top

def pearson_sim(a, b):
    #pearson = cosine on mean-centred vectors -> accounts for rating scale differences per user
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = np.linalg.norm(a_c) * np.linalg.norm(b_c)
    if denom == 0:
        return 0.0
    return float(np.dot(a_c, b_c) / denom)

def user_based_cf(user_id, user_item, df, movies, n=10, k=20, metric="cosine"):
    if user_id not in user_item.index:
        return pd.DataFrame()
    u_vec = user_item.loc[user_id].values
    if metric == "pearson":
        #compute pearson vs every other user -> slower but more fair across different rating scales
        sims = np.array([pearson_sim(u_vec, user_item.loc[uid].values)
                         for uid in user_item.index])
    else:
        sims = cosine_similarity(u_vec.reshape(1,-1), user_item.values)[0]
    sim_df = pd.Series(sims, index=user_item.index).drop(user_id).nlargest(k)
    seen = set(df[df["userId"]==user_id]["movieId"])
    neighbor_ratings = user_item.loc[sim_df.index]
    #weighted avg: neighbors closer in taste count more
    weighted = neighbor_ratings.T.dot(sim_df) / (sim_df.sum() + 1e-9)
    unseen = weighted.drop(index=[m for m in seen if m in weighted.index], errors="ignore")
    top_ids = unseen.nlargest(n).index.tolist()
    return movies[movies["movieId"].isin(top_ids)][["movieId","title","genres"]].assign(
        score=lambda x: x["movieId"].map(dict(zip(top_ids, range(n,0,-1))))
    ).sort_values("score", ascending=False)

def item_based_cf(movie_id, user_item, movies, n=10, metric="cosine"):
    if movie_id not in user_item.columns:
        return pd.DataFrame()
    item_vec = user_item[movie_id].values
    if metric == "pearson":
        #pearson on item vectors = correlation over users who rated both items
        sims = np.array([pearson_sim(item_vec, user_item[mid].values)
                         for mid in user_item.columns])
    else:
        sims = cosine_similarity(item_vec.reshape(1,-1), user_item.values.T)[0]
    sim_series = pd.Series(sims, index=user_item.columns).drop(movie_id)
    top_ids = sim_series.nlargest(n).index.tolist()
    return movies[movies["movieId"].isin(top_ids)][["movieId","title","genres"]]

def content_based(movie_id, movies_ext, tfidf_matrix, n=10):
    idx_map = pd.Series(movies_ext.index, index=movies_ext["movieId"])
    if movie_id not in idx_map:
        return pd.DataFrame()
    idx = idx_map[movie_id]
    sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix)[0]
    sim_series = pd.Series(sims, index=movies_ext["movieId"]).drop(movie_id)
    top_ids = sim_series.nlargest(n).index.tolist()
    return movies_ext[movies_ext["movieId"].isin(top_ids)][["movieId","title","genres"]]

def cbf_user_profile(user_id, df, movies_ext, tfidf_matrix, n=10):
    user_ratings = df[df["userId"]==user_id][["movieId","rating"]]
    if user_ratings.empty:
        return pd.DataFrame()
    avg = user_ratings["rating"].mean()
    idx_map = pd.Series(movies_ext.index, index=movies_ext["movieId"])
    valid = user_ratings[user_ratings["movieId"].isin(idx_map.index)]
    indices = idx_map[valid["movieId"]].values
    #centering ratings so below-avg movies get negative weight in profile
    weights = (valid["rating"].values - avg)
    profile = weights @ tfidf_matrix[indices].toarray()
    sims = cosine_similarity([profile], tfidf_matrix)[0]
    all_scores = pd.Series(sims, index=movies_ext["movieId"])
    seen = set(valid["movieId"])
    unseen = all_scores.drop(index=[m for m in seen if m in all_scores.index], errors="ignore")
    top_ids = unseen.nlargest(n).index.tolist()
    return movies_ext[movies_ext["movieId"].isin(top_ids)][["movieId","title","genres"]]

def matrix_factorization_recs(user_id, user_item, movies, df, n=10, n_components=50):
    #truncatedsvd approximates R ~ P * Q.T using latent factors
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = svd.fit_transform(csr_matrix(user_item.values))
    item_factors = svd.components_.T
    if user_id not in user_item.index:
        return pd.DataFrame()
    u_idx = user_item.index.get_loc(user_id)
    scores = user_factors[u_idx] @ item_factors.T
    all_scores = pd.Series(scores, index=user_item.columns)
    seen = set(df[df["userId"]==user_id]["movieId"])
    unseen = all_scores.drop(index=[m for m in seen if m in all_scores.index], errors="ignore")
    top_ids = unseen.nlargest(n).index.tolist()
    return movies[movies["movieId"].isin(top_ids)][["movieId","title","genres"]]

def diversity(movie_ids, movies_ext, tfidf_matrix):
    #diversity = 1 - avg pairwise cosine sim -> higher = more varied recommendations
    idx_map = pd.Series(movies_ext.index, index=movies_ext["movieId"])
    valid_ids = [m for m in movie_ids if m in idx_map.index]
    if len(valid_ids) < 2:
        return 0.0
    vecs = tfidf_matrix[idx_map[valid_ids].values]
    sims = cosine_similarity(vecs)
    n = len(valid_ids)
    #avg of upper triangle only -> avoid self-sim + double-counting pairs
    upper = sims[np.triu_indices(n, k=1)]
    return float(1 - upper.mean())

def novelty(movie_ids, df, n_users):
    #novelty = -log2(p) where p = fraction of users who rated the movie
    #popular items -> low novelty, niche items -> high novelty
    rating_counts = df.groupby("movieId")["userId"].count()
    scores = []
    for mid in movie_ids:
        count = rating_counts.get(mid, 1)
        p = count / n_users
        scores.append(-np.log2(p))
    return float(np.mean(scores)) if scores else 0.0

def ndcg_at_k(recommended_ids, relevant_ids, k=10):
    #ndcg rewards hits ranked higher -> position matters unlike precision
    #binary relevance: 1 if item is relevant, 0 otherwise
    rec = recommended_ids[:k]
    gains = [1 if mid in relevant_ids else 0 for mid in rec]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    #ideal dcg = all relevant items at top positions
    ideal = sum(1 / np.log2(i + 2) for i in range(min(len(relevant_ids), k)))
    return float(dcg / ideal) if ideal > 0 else 0.0


st.title("🎬 MovieLens Recommender System")
st.caption("Esade - Recommender Systems | Individual Project")

with st.spinner("Loading MovieLens data..."):
    try:
        ratings, movies, tags = load_data()
        df, movies_ext = preprocess(ratings, movies, tags)
        tfidf_matrix = build_tfidf(movies_ext)
        user_item = build_user_item(df)
        st.success(f"Loaded {len(ratings):,} ratings · {len(movies):,} movies · {df['userId'].nunique():,} users")
    except Exception as e:
        st.error(f"Could not load data: {e}")
        st.stop()

tabs = st.tabs(["📊 EDA", "🏆 Non-Personalised", "👥 Collaborative Filtering", "🏷️ Content-Based", "🔢 Matrix Factorisation", "📐 Evaluation", "👤 User Comparison"])

with tabs[0]:
    st.header("Exploratory Data Analysis")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ratings", f"{len(ratings):,}")
    c2.metric("Movies", f"{len(movies):,}")
    c3.metric("Users", f"{ratings['userId'].nunique():,}")
    c4.metric("Avg Rating", f"{ratings['rating'].mean():.2f}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Rating Distribution")
        hist = ratings["rating"].value_counts().sort_index()
        st.bar_chart(hist)

    with col2:
        st.subheader("Top Genres by Number of Movies")
        genre_counts = (movies["genres"].str.split("|")
                        .explode()
                        .value_counts()
                        .head(15))
        st.bar_chart(genre_counts)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Ratings per User")
        user_counts = ratings.groupby("userId").size()
        bins = pd.cut(user_counts, bins=[0,20,50,100,200,500,1000,9999])
        st.bar_chart(bins.value_counts().sort_index().rename(str))

    with col4:
        #long tail: most movies have very few ratings -> this is why popularity bias matters
        st.subheader("Avg Rating by Genre")
        genre_ratings = (df.assign(genre=df["genres"].str.split("|"))
                           .explode("genre")
                           .groupby("genre")["rating"]
                           .mean()
                           .sort_values(ascending=False)
                           .drop("(no genres listed)", errors="ignore"))
        st.bar_chart(genre_ratings)

    st.subheader("Sparsity")
    n_users = ratings["userId"].nunique()
    n_movies = movies["movieId"].nunique()
    sparsity = 1 - len(ratings) / (n_users * n_movies)
    st.info(f"Matrix sparsity: **{sparsity:.2%}** - only {1-sparsity:.2%} of all user-movie pairs have a rating.")

with tabs[1]:
    st.header("Non-Personalised Recommendations")

    all_genres = sorted(set(
        movies["genres"].str.split("|").explode().dropna().unique()
    ) - {"(no genres listed)"})

    col_method, col_genre = st.columns([2,1])
    with col_method:
        method = st.radio("Method", ["popularity", "bayesian", "highest_rated"], horizontal=True,
                          format_func=lambda x: {"popularity":"Most Rated","bayesian":"Bayesian Average","highest_rated":"Highest Rated (min 50 ratings)"}[x])
    with col_genre:
        genre_filter = st.selectbox("Filter by genre", ["All"] + all_genres)

    n_recs = st.slider("Number of recommendations", 5, 20, 10)

    result = non_personalized_recs(df, movies, method=method, n=n_recs, genre=genre_filter)

    show_cols = ["title","genres","count","mean"]
    if method == "bayesian" and "bayes" in result.columns:
        show_cols = ["title","genres","count","mean","bayes"]

    st.dataframe(result[show_cols].rename(
        columns={"count":"# Ratings","mean":"Avg Rating","bayes":"Bayesian Score"}
    ).reset_index(drop=True), use_container_width=True)

    st.info("""
**Methods:**
- **Most Rated**: pure popularity, biased toward mainstream content
- **Bayesian Average**: shrinks rating toward global mean when vote count is low
- **Highest Rated**: filters by min 50 votes to reduce noise from obscure films
""")

with tabs[2]:
    st.header("Collaborative Filtering")
    cf_type = st.radio("Type", ["User-Based kNN", "Item-Based kNN"], horizontal=True)
    sim_metric = st.radio("Similarity metric", ["cosine", "pearson"], horizontal=True,
                          format_func=lambda x: {"cosine":"Cosine","pearson":"Pearson"}[x])

    if cf_type == "User-Based kNN":
        user_id = st.number_input("User ID", min_value=int(ratings["userId"].min()),
                                  max_value=int(ratings["userId"].max()), value=1, step=1)
        k = st.slider("k neighbours", 5, 50, 20)
        n_cf = st.slider("Recommendations", 5, 20, 10)
        if st.button("Get User-Based Recommendations"):
            with st.spinner("Computing user similarities..."):
                recs = user_based_cf(int(user_id), user_item, df, movies, n=n_cf, k=k, metric=sim_metric)
            if recs.empty:
                st.warning("No recommendations found for this user.")
            else:
                st.subheader(f"Top {n_cf} recommendations for User {user_id} ({sim_metric})")
                st.dataframe(recs[["title","genres"]].reset_index(drop=True), use_container_width=True)
                user_hist = df[df["userId"]==user_id].nlargest(10,"rating")[["movieId","rating"]].merge(movies[["movieId","title"]], on="movieId")
                with st.expander("User's top-rated movies"):
                    st.dataframe(user_hist.reset_index(drop=True))
    else:
        movie_title = st.selectbox("Select a movie", movies["title"].sort_values().tolist())
        movie_id = int(movies[movies["title"]==movie_title]["movieId"].iloc[0])
        n_cf = st.slider("Recommendations", 5, 20, 10)
        if st.button("Get Item-Based Recommendations"):
            with st.spinner("Computing item similarities..."):
                recs = item_based_cf(movie_id, user_item, movies, n=n_cf, metric=sim_metric)
            st.subheader(f"Movies similar to '{movie_title}' ({sim_metric})")
            st.dataframe(recs[["title","genres"]].reset_index(drop=True), use_container_width=True)

    st.info("""
**Similarity metrics:**
- **Cosine:** angle between rating vectors - fast + standard baseline
- **Pearson:** cosine on mean-centred vectors - corrects for users who consistently rate high/low

**User-based:** find k most similar users, take weighted avg of their ratings.
**Item-based:** find most similar items based on who rated them similarly.
""")

with tabs[3]:
    st.header("Content-Based Filtering")
    cbf_mode = st.radio("Mode", ["By Movie (item-to-item)", "By User Profile"], horizontal=True)

    if cbf_mode == "By Movie (item-to-item)":
        movie_title_cbf = st.selectbox("Select a movie", movies["title"].sort_values().tolist(), key="cbf_movie")
        movie_id_cbf = int(movies[movies["title"]==movie_title_cbf]["movieId"].iloc[0])
        n_cbf = st.slider("Recommendations", 5, 20, 10, key="cbf_n")
        if st.button("Find Similar Movies (Content)"):
            with st.spinner("Computing TF-IDF cosine similarity..."):
                recs = content_based(movie_id_cbf, movies_ext, tfidf_matrix, n=n_cbf)
            st.subheader(f"Content-similar to '{movie_title_cbf}'")
            st.dataframe(recs[["title","genres"]].reset_index(drop=True), use_container_width=True)
    else:
        user_id_cbf = st.number_input("User ID", min_value=int(ratings["userId"].min()),
                                      max_value=int(ratings["userId"].max()), value=1, step=1, key="cbf_user")
        n_cbf2 = st.slider("Recommendations", 5, 20, 10, key="cbf_n2")
        if st.button("Get CBF Recommendations for User"):
            with st.spinner("Building user content profile..."):
                recs = cbf_user_profile(int(user_id_cbf), df, movies_ext, tfidf_matrix, n=n_cbf2)
            if recs.empty:
                st.warning("No recommendations.")
            else:
                st.subheader(f"CBF recommendations for User {user_id_cbf}")
                st.dataframe(recs[["title","genres"]].reset_index(drop=True), use_container_width=True)

    st.info("""
**Item features:** genres + user tags -> TF-IDF vectors (500 features).
**Item-to-item:** cosine similarity between TF-IDF vectors.
**User profile:** profile(u) = sum of (rating - avg) * tfidf(item) -> cosine vs all unseen items.
""")

with tabs[4]:
    st.header("Matrix Factorisation (SVD)")
    user_id_mf = st.number_input("User ID", min_value=int(ratings["userId"].min()),
                                  max_value=int(ratings["userId"].max()), value=1, step=1, key="mf_user")
    n_components = st.slider("Latent factors", 10, 100, 50)
    n_mf = st.slider("Recommendations", 5, 20, 10, key="mf_n")
    if st.button("Get MF Recommendations"):
        with st.spinner("Running TruncatedSVD..."):
            recs = matrix_factorization_recs(int(user_id_mf), user_item, movies, df, n=n_mf, n_components=n_components)
        if recs.empty:
            st.warning("No recommendations.")
        else:
            st.subheader(f"MF recommendations for User {user_id_mf}")
            st.dataframe(recs[["title","genres"]].reset_index(drop=True), use_container_width=True)

    st.info("""
**Method:** TruncatedSVD on the user-item matrix. Decomposes R ~ P * Q.T
Predicted score: p_u dot q_i
**Limitations:** cold-start problem, latent factors are not interpretable, tends to favour popular items.
""")

with tabs[5]:
    st.header("Evaluation")
    st.markdown("""
### Methodology
- **Temporal split** (80/20 per user): prevents data leakage
- **Held-out set:** movies rated >= 4 in the test period (considered relevant)
- **Accuracy metrics:** Precision@10, Recall@10
- **Beyond-accuracy metrics:** Diversity (intra-list variety), Novelty (inverse popularity)
- **Baselines:** Most Popular + Random
""")

    if st.button("Run Evaluation (sample 50 users)"):
        results = {}
        n_total_users = df["userId"].nunique()
        users_eval = df["userId"].value_counts()[lambda x: x >= 20].index[:50].tolist()
        df_sorted = df.sort_values("timestamp")

        with st.spinner("Evaluating Most Popular..."):
            pop_recs_ids = list(non_personalized_recs(df, movies, method="popularity", n=10)["movieId"])
            pop_ids = set(pop_recs_ids)
            precs, recs_r, ndcgs = [], [], []
            div_scores, nov_scores = [], []
            for uid in users_eval:
                udata = df_sorted[df_sorted["userId"]==uid]
                split = int(len(udata)*0.8)
                test_liked = set(udata.iloc[split:][udata.iloc[split:]["rating"]>=4]["movieId"])
                if not test_liked: continue
                hits = pop_ids & test_liked
                precs.append(len(hits)/10)
                recs_r.append(len(hits)/len(test_liked))
                ndcgs.append(ndcg_at_k(pop_recs_ids, test_liked))
                div_scores.append(diversity(pop_recs_ids, movies_ext, tfidf_matrix))
                nov_scores.append(novelty(pop_recs_ids, df, n_total_users))
            results["Most Popular"] = {
                "Precision@10": np.mean(precs), "Recall@10": np.mean(recs_r),
                "NDCG@10": np.mean(ndcgs),
                "Diversity": np.mean(div_scores), "Novelty": np.mean(nov_scores)
            }

        with st.spinner("Evaluating User-Based CF..."):
            precs_cf, recs_cf, ndcgs_cf, div_cf, nov_cf = [], [], [], [], []
            for uid in users_eval:
                udata = df_sorted[df_sorted["userId"]==uid]
                split = int(len(udata)*0.8)
                test_liked = set(udata.iloc[split:][udata.iloc[split:]["rating"]>=4]["movieId"])
                if not test_liked: continue
                try:
                    rec = user_based_cf(uid, user_item, df, movies, n=10)
                    if rec.empty: continue
                    rec_ids = list(rec["movieId"])
                    hits = set(rec_ids) & test_liked
                    precs_cf.append(len(hits)/10)
                    recs_cf.append(len(hits)/len(test_liked))
                    ndcgs_cf.append(ndcg_at_k(rec_ids, test_liked))
                    div_cf.append(diversity(rec_ids, movies_ext, tfidf_matrix))
                    nov_cf.append(novelty(rec_ids, df, n_total_users))
                except: continue
            results["User-Based CF"] = {
                "Precision@10": np.mean(precs_cf), "Recall@10": np.mean(recs_cf),
                "NDCG@10": np.mean(ndcgs_cf),
                "Diversity": np.mean(div_cf), "Novelty": np.mean(nov_cf)
            }

        with st.spinner("Evaluating Content-Based..."):
            precs_cbf, recs_cbf, ndcgs_cbf, div_cbf, nov_cbf = [], [], [], [], []
            for uid in users_eval:
                udata = df_sorted[df_sorted["userId"]==uid]
                split = int(len(udata)*0.8)
                test_liked = set(udata.iloc[split:][udata.iloc[split:]["rating"]>=4]["movieId"])
                if not test_liked: continue
                try:
                    rec = cbf_user_profile(uid, df, movies_ext, tfidf_matrix, n=10)
                    if rec.empty: continue
                    rec_ids = list(rec["movieId"])
                    hits = set(rec_ids) & test_liked
                    precs_cbf.append(len(hits)/10)
                    recs_cbf.append(len(hits)/len(test_liked))
                    ndcgs_cbf.append(ndcg_at_k(rec_ids, test_liked))
                    div_cbf.append(diversity(rec_ids, movies_ext, tfidf_matrix))
                    nov_cbf.append(novelty(rec_ids, df, n_total_users))
                except: continue
            results["Content-Based"] = {
                "Precision@10": np.mean(precs_cbf), "Recall@10": np.mean(recs_cbf),
                "NDCG@10": np.mean(ndcgs_cbf),
                "Diversity": np.mean(div_cbf), "Novelty": np.mean(nov_cbf)
            }

        with st.spinner("Evaluating Matrix Factorisation..."):
            svd = TruncatedSVD(n_components=50, random_state=42)
            uf = svd.fit_transform(csr_matrix(user_item.values))
            itf = svd.components_.T
            precs_mf, recs_mf, ndcgs_mf, div_mf, nov_mf = [], [], [], [], []
            for uid in users_eval:
                udata = df_sorted[df_sorted["userId"]==uid]
                split = int(len(udata)*0.8)
                test_liked = set(udata.iloc[split:][udata.iloc[split:]["rating"]>=4]["movieId"])
                if not test_liked or uid not in user_item.index: continue
                u_idx = user_item.index.get_loc(uid)
                scores = pd.Series(uf[u_idx] @ itf.T, index=user_item.columns)
                seen = set(udata.iloc[:split]["movieId"])
                unseen = scores.drop(index=[m for m in seen if m in scores.index], errors="ignore")
                rec_ids = list(unseen.nlargest(10).index)
                hits = set(rec_ids) & test_liked
                precs_mf.append(len(hits)/10)
                recs_mf.append(len(hits)/len(test_liked))
                ndcgs_mf.append(ndcg_at_k(rec_ids, test_liked))
                div_mf.append(diversity(rec_ids, movies_ext, tfidf_matrix))
                nov_mf.append(novelty(rec_ids, df, n_total_users))
            results["Matrix Factorisation"] = {
                "Precision@10": np.mean(precs_mf), "Recall@10": np.mean(recs_mf),
                "NDCG@10": np.mean(ndcgs_mf),
                "Diversity": np.mean(div_mf), "Novelty": np.mean(nov_mf)
            }

        #random baseline -> lower bound for accuracy but upper bound for novelty/diversity
        all_movie_ids = movies["movieId"].tolist()
        precs_r, recs_r2, ndcgs_r, div_rand, nov_rand = [], [], [], [], []
        for uid in users_eval:
            udata = df_sorted[df_sorted["userId"]==uid]
            split = int(len(udata)*0.8)
            test_liked = set(udata.iloc[split:][udata.iloc[split:]["rating"]>=4]["movieId"])
            if not test_liked: continue
            rand_ids = list(np.random.choice(all_movie_ids, 10, replace=False))
            hits = set(rand_ids) & test_liked
            precs_r.append(len(hits)/10)
            recs_r2.append(len(hits)/len(test_liked))
            ndcgs_r.append(ndcg_at_k(rand_ids, test_liked))
            div_rand.append(diversity(rand_ids, movies_ext, tfidf_matrix))
            nov_rand.append(novelty(rand_ids, df, n_total_users))
        results["Random Baseline"] = {
            "Precision@10": np.mean(precs_r), "Recall@10": np.mean(recs_r2),
            "NDCG@10": np.mean(ndcgs_r),
            "Diversity": np.mean(div_rand), "Novelty": np.mean(nov_rand)
        }

        eval_df = pd.DataFrame(results).T.round(4)
        st.subheader("Results")
        st.dataframe(eval_df, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Precision@10**")
            st.bar_chart(eval_df["Precision@10"])
        with col_b:
            st.markdown("**Diversity**")
            st.bar_chart(eval_df["Diversity"])

        st.info("""
**Diversity:** 1 - avg pairwise cosine similarity between recommended items (tfidf vectors). higher = more varied list.
**Novelty:** avg -log2(p) where p = fraction of users who rated the item. higher = more niche recommendations.
Accuracy + diversity trade-off: popularity-based recs score low on novelty because they push the same mainstream films.
""")

    st.markdown("""
### Pitfalls avoided
- temporal split (not random) to prevent leakage
- never recommend already-seen items
- compared against baselines (random + popularity)
- accuracy + beyond-accuracy metrics together give a fuller picture of recommendation quality
""")

with tabs[6]:
    st.header("User Comparison")
    st.markdown("Top-5 recommendations for three different users across all algorithms - shows how each method personalises differently.")

    #three users with enough ratings to get meaningful recs
    compare_users = [1, 50, 200]

    #precompute svd once for mf recs in this tab
    svd_comp = TruncatedSVD(n_components=50, random_state=42)
    uf_comp = svd_comp.fit_transform(csr_matrix(user_item.values))
    itf_comp = svd_comp.components_.T

    for uid in compare_users:
        st.subheader(f"User {uid}")

        #show what this user has rated + liked
        user_hist = df[df["userId"]==uid].nlargest(5, "rating")[["movieId","rating"]].merge(movies[["movieId","title","genres"]], on="movieId")
        with st.expander(f"User {uid} - top rated movies"):
            st.dataframe(user_hist.reset_index(drop=True), use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("**User-Based CF**")
            try:
                r = user_based_cf(uid, user_item, df, movies, n=5)
                st.dataframe(r[["title"]].reset_index(drop=True), use_container_width=True)
            except:
                st.write("n/a")

        with col2:
            st.markdown("**Content-Based**")
            try:
                r = cbf_user_profile(uid, df, movies_ext, tfidf_matrix, n=5)
                st.dataframe(r[["title"]].reset_index(drop=True), use_container_width=True)
            except:
                st.write("n/a")

        with col3:
            st.markdown("**Matrix Factorisation**")
            try:
                if uid in user_item.index:
                    u_idx = user_item.index.get_loc(uid)
                    sc = pd.Series(uf_comp[u_idx] @ itf_comp.T, index=user_item.columns)
                    seen = set(df[df["userId"]==uid]["movieId"])
                    unseen = sc.drop(index=[m for m in seen if m in sc.index], errors="ignore")
                    top_ids = unseen.nlargest(5).index.tolist()
                    r = movies[movies["movieId"].isin(top_ids)][["title"]]
                    st.dataframe(r.reset_index(drop=True), use_container_width=True)
                else:
                    st.write("n/a")
            except:
                st.write("n/a")

        with col4:
            st.markdown("**Most Popular**")
            pop = non_personalized_recs(df, movies, method="popularity", n=5)[["title"]]
            st.dataframe(pop.reset_index(drop=True), use_container_width=True)

        st.divider()
