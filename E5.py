from sklearn.cluster import KMeans

from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from methods import * 
def main():
    # Random Seed
    gtid=903789757
    # Load dataset
    # digits = load_digits()
    # X = digits.data
    # y = digits.target

    X,y,_,_,= loan_datasets()
    # Standardize the dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply clustering algorithms
    kmeans = KMeans(n_clusters=3, random_state=gtid)
    em = GaussianMixture(n_components=3, random_state=gtid)

    kmeans_clusters = kmeans.fit_predict(X_scaled)
    em_clusters = em.fit_predict(X_scaled)

    # Add cluster labels as new features to the dataset
    X_kmeans = np.column_stack((X_scaled, kmeans_clusters))
    X_em = np.column_stack((X_scaled, em_clusters))

    # Split the datasets into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=gtid)
    X_train_kmeans, X_test_kmeans, y_train_kmeans, y_test_kmeans = train_test_split(X_kmeans, y, test_size=0.3, random_state=gtid)
    X_train_em, X_test_em, y_train_em, y_test_em = train_test_split(X_em, y, test_size=0.3, random_state=gtid)

    nn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=gtid)
    nn.fit(X_train, y_train)
    y_pred = nn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Train and evaluate on KMeans augmented data
    nn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=gtid)
    nn.fit(X_train_kmeans, y_train_kmeans)
    y_pred_kmeans = nn.predict(X_test_kmeans)
    accuracy_kmeans = accuracy_score(y_test_kmeans, y_pred_kmeans)

    # Train and evaluate on EM augmented data
    nn2 = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=gtid)
    nn2.fit(X_train_em, y_train_em)
    y_pred_em = nn2.predict(X_test_em)
    accuracy_em = accuracy_score(y_test_em, y_pred_em)

    # Print results
    print('original:',accuracy)
    print('k means:',accuracy_kmeans)
    print('em: ', accuracy_em)


if __name__ == '__main__':
    main()