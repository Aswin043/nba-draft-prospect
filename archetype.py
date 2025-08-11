import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

DEFAULT_NUMERIC = ["fg_pct","fg3_pct","ft_pct","mp_per_g","pts_per_g","trb_per_g","ast_per_g"]

def load_df(path: str):
    df = pd.read_csv(path)
    missing = [c for c in DEFAULT_NUMERIC if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df

def apply_weights(X: pd.DataFrame, weights: dict):
    W = X.copy()
    for col, w in weights.items():
        if col in W.columns:
            W[col] = W[col] * float(w)
    return W

def compute_pca(X: pd.DataFrame, n_components=2, random_state=0):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=n_components, random_state=random_state)
    Z = pca.fit_transform(Xs)
    return Z, pca, scaler

def clusterize(X: pd.DataFrame, algo: str = "kmeans", k: int = 4, random_state=0):
    if algo == "kmeans":
        model = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        labels = model.fit_predict(X)
        return labels, model
    elif algo == "gmm":
        model = GaussianMixture(n_components=k, random_state=random_state)
        labels = model.fit_predict(X)
        return labels, model
    else:
        raise ValueError("algo must be 'kmeans' or 'gmm'")

def compute_neighbors(X: pd.DataFrame, k: int = 6):
    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X)
    return distances, indices

def build(path_in: str, path_out: str, algo: str, k: int, weights: dict, neighbors_k: int):
    df = load_df(path_in)
    X = df[DEFAULT_NUMERIC].copy()
    X = X.fillna(X.median())
    Xw = apply_weights(X, weights)

    Z, pca, scaler = compute_pca(Xw, n_components=2)
    Xs = scaler.transform(Xw)
    cluster_labels, cluster_model = clusterize(Xs, algo=algo, k=k)

    dists, idxs = compute_neighbors(Xs, k=neighbors_k)

    out = df.copy()
    out["PC1"] = Z[:,0]
    out["PC2"] = Z[:,1]
    out["cluster"] = cluster_labels

    players = out["Player"].astype(str).tolist() if "Player" in out.columns else [str(i) for i in range(len(out))]
    rows = []
    for i, name in enumerate(players):
        neighs = [(players[j], float(dists[i][pos])) for pos, j in enumerate(idxs[i]) if j != i]
        top5 = neighs[:5]
        rows.append({
            "Player": name,
            **{f"NN{r+1}": p for r, (p, _) in enumerate(top5)},
            **{f"NN{r+1}_dist": d for r, (_, d) in enumerate(top5)}
        })
    nn_df = pd.DataFrame(rows)

    out.to_csv(path_out, index=False)
    nn_df.to_csv(path_out.replace(".csv", "_neighbors.csv"), index=False)

    meta = {
        "explained_variance_ratio": getattr(pca, "explained_variance_ratio_", None).tolist() if hasattr(pca, "explained_variance_ratio_") else None,
        "algo": algo,
        "k": k,
        "weights": weights,
        "neighbors_k": neighbors_k,
        "features": DEFAULT_NUMERIC,
    }
    with open(path_out.replace(".csv", "_meta.json"), "w", encoding="utf-8") as f:
        import json; json.dump(meta, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="archetypes_prepared.csv")
    parser.add_argument("--algo", type=str, default="kmeans", choices=["kmeans","gmm"])
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--neighbors_k", type=int, default=6)
    parser.add_argument("--weights", type=str, default="{}")
    args = parser.parse_args()
    import json as _json
    weights = {}
    if args.weights:
        weights = _json.loads(args.weights)
    build(args.input, args.output, args.algo, args.k, weights, args.neighbors_k)
