# import numpy as np
# from sklearn.datasets import load_iris, load_diabetes
# from sklearn.mixture import GaussianMixture
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import adjusted_rand_score, silhouette_score
# from sklearn.random_projection import GaussianRandomProjection
# from sklearn.decomposition import PCA, FastICA
# import matplotlib.pyplot as plt
# import os


# def apply_randomized_projections(X, n_components,seed=0):
#     rp = GaussianRandomProjection(n_components=n_components, random_state=seed)
#     return rp.fit_transform(X)

# def apply_pca(X, n_components,seed=0):
#     pca = PCA(n_components=n_components, random_state=seed)
#     return pca.fit_transform(X)

# def apply_ica(X, n_components,seed=0):
#     ica = FastICA(n_components=n_components, random_state=seed)
#     return ica.fit_transform(X)

# from sklearn.manifold import LocallyLinearEmbedding
# from sklearn.neighbors import NearestNeighbors
# def apply_lle(X, n_components, dataset_name, seed=0):
#     lle = LocallyLinearEmbedding(n_neighbors=30, n_components=n_components, random_state=seed)
#     X_lle = lle.fit_transform(X)
    
#     # 计算重构误差
#     nbrs = NearestNeighbors(n_neighbors=30).fit(X_lle)
#     distances, indices = nbrs.kneighbors(X_lle)
    
#     X_reconstructed = np.zeros_like(X)
#     for i in range(len(X)):
#         weights = np.exp(-distances[i] ** 2)
#         weights /= np.sum(weights)
#         X_reconstructed[i] = np.dot(weights, X[indices[i]])
    
#     reconstruction_error = np.mean(np.linalg.norm(X - X_reconstructed, axis=1))
    
#     print(f"LLE on {dataset_name}:")
#     print(f"Reconstruction Error: {reconstruction_error:.2f}\n")
    
#     return X_lle

# def apply_em(X, y, n_clusters, dataset_name, dr_method,seed=0):
#     # Apply Gaussian Mixture Model
#     gmm = GaussianMixture(n_components=n_clusters, random_state=seed)
#     y_pred = gmm.fit_predict(X)
    
#     # Evaluate the clustering
#     ari = adjusted_rand_score(y, y_pred)
#     silhouette_avg = silhouette_score(X, y_pred)
    
#     print(f"EM on {dataset_name} with {dr_method}:")
#     print(f"Adjusted Rand Index: {ari:.2f}")
#     print(f"Silhouette Score: {silhouette_avg:.2f}\n")
    
#     return y_pred, ari, silhouette_avg

# def apply_kmeans(X, y, n_clusters, dataset_name, dr_method,seed=0):
#     # Apply K-Means
#     kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
#     y_pred = kmeans.fit_predict(X)
    
#     # Evaluate the clustering
#     ari = adjusted_rand_score(y, y_pred)
#     silhouette_avg = silhouette_score(X, y_pred)
    
#     print(f"K-Means on {dataset_name} with {dr_method}:")
#     print(f"Adjusted Rand Index: {ari:.2f}")
#     print(f"Silhouette Score: {silhouette_avg:.2f}\n")
    
#     return y_pred, ari, silhouette_avg



from methods import * 
def main():
   # Random Seed
    gtid=903789757
    # Load datasets
    X_iris,y_iris,X_diabetes,y_diabetes = loan_datasets()

    # Number of components (for iris, let's keep 2, and for diabetes, let's keep 10)
    n_components_iris = 2
    n_components_diabetes = 5

    # Number of clusters (3 for iris and 2 for diabetes, as an example)
    n_clusters_iris = 3
    n_clusters_diabetes = 2

    # Dimensionality reduction methods
    dim_reduction_methods = {
        "Randomized Projections": apply_randomized_projections,
        "PCA": apply_pca,
        "ICA": apply_ica,
        "LLE":apply_lle
    }

    # Datasets
    datasets = {
        "Iris": (X_iris, y_iris, n_components_iris, n_clusters_iris),
        "Diabetes": (X_diabetes, y_diabetes, n_components_diabetes, n_clusters_diabetes)
    }
    e3(dim_reduction_methods,datasets,gtid)
    

if __name__ == '__main__':
    main()