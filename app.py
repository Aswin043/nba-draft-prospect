
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

st.set_page_config(page_title="NBA Draft 2025 – Archetypes & Similarity", layout="wide")

DEFAULT_NUMERIC = ["fg_pct","fg3_pct","ft_pct","mp_per_g","pts_per_g","trb_per_g","ast_per_g"]

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

def apply_weights(X, weights):
    W = X.copy()
    for col, w in weights.items():
        if col in W.columns:
            W[col] = W[col] * float(w)
    return W

def standardize_and_pca(X, n_components=2):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=n_components, random_state=0)
    Z = pca.fit_transform(Xs)
    return Z, pca, scaler

def clusterize(Xs, algo="kmeans", k=4):
    if algo == "kmeans":
        model = KMeans(n_clusters=k, n_init=20, random_state=0)
        labels = model.fit_predict(Xs)
    else:
        model = GaussianMixture(n_components=k, random_state=0)
        labels = model.fit_predict(Xs)
    return labels, model

def nearest_neighbors(Xs, k=6):
    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nbrs.fit(Xs)
    dists, idxs = nbrs.kneighbors(Xs)
    return dists, idxs

st.title("NBA Draft 2025 – Player Archetypes & Similarity Search")

st.sidebar.header("Data & Settings")
csv_path = st.sidebar.text_input("CSV path", "nba_draft_2025_final.csv")
algo = st.sidebar.selectbox("Clustering algorithm", ["kmeans","gmm"], index=0)
k = st.sidebar.slider("Number of clusters (k)", 2, 8, 4)

st.sidebar.header("Feature Weights")
weights = {}
for col in DEFAULT_NUMERIC:
    weights[col] = st.sidebar.slider(col, 0.5, 2.0, 1.0, 0.05)

df = load_csv(csv_path)
if not set(DEFAULT_NUMERIC).issubset(df.columns):
    st.error(f"CSV must contain: {DEFAULT_NUMERIC}")
    st.stop()

X = df[DEFAULT_NUMERIC].copy().fillna(df[DEFAULT_NUMERIC].median())
Xw = apply_weights(X, weights)

Z, pca, scaler = standardize_and_pca(Xw)
Xs = scaler.transform(Xw)

labels, model = clusterize(Xs, algo=algo, k=k)
dists, idxs = nearest_neighbors(Xs, k=6)

left, right = st.columns([1,1])

with right:
    fig = plt.figure(figsize=(7,5))
    plt.scatter(Z[:,0], Z[:,1])
    if "Player" in df.columns:
        for i, name in enumerate(df["Player"].astype(str).tolist()):
            plt.annotate(name, (Z[i,0], Z[i,1]), fontsize=8, alpha=0.7)
    plt.title("PCA (2D) – standardized weighted stats")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    st.pyplot(fig)

with left:
    st.subheader("Clustered Players")
    out = df.copy()
    out["PC1"] = Z[:,0]
    out["PC2"] = Z[:,1]
    out["cluster"] = labels
    st.dataframe(out)

st.markdown("---")
st.header("Similarity Search")
player_names = df["Player"].astype(str).tolist() if "Player" in df.columns else [str(i) for i in range(len(df))]
pick = st.selectbox("Pick a player", player_names, index=0)

if pick:
    i = player_names.index(pick)
    neighs = [(player_names[j], float(dists[i][pos])) for pos, j in enumerate(idxs[i]) if j != i][:5]
    st.subheader(f"Nearest Neighbors to {pick}")
    st.table(pd.DataFrame(neighs, columns=["Similar Player","Distance"]))

st.markdown("---")
st.caption("Tip: Adjust feature weights to emphasize different roles (e.g., bump `ast_per_g` for guards, `trb_per_g` for bigs).")
