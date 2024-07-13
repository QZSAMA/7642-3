import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.manifold import LocallyLinearEmbedding
from datetime import datetime

from methods import * 
def main():
    # Random Seed
    gtid=903789757
    # Load dataset
    X,y,_,_,= loan_datasets()
    # X = diabetes.data
    # y = digits.target

    # Standardize the dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Dimensionality reduction techniques
    pca = PCA(n_components=2)
    ica = FastICA(n_components=2, random_state=gtid)
    rp = GaussianRandomProjection(n_components=2, random_state=gtid)
    lle =LocallyLinearEmbedding(n_neighbors=2, random_state=gtid)
    X_pca = pca.fit_transform(X_scaled)
    X_ica = ica.fit_transform(X_scaled)
    X_rp = rp.fit_transform(X_scaled)
    X_lle = lle.fit_transform(X_scaled)
    # Split the datasets into training and testing sets
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X_scaled, y, test_size=0.3, random_state=gtid)
    X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.3, random_state=gtid)
    X_train_ica, X_test_ica, y_train_ica, y_test_ica = train_test_split(X_ica, y, test_size=0.3, random_state=gtid)
    X_train_rp, X_test_rp, y_train_rp, y_test_rp = train_test_split(X_rp, y, test_size=0.3, random_state=gtid)
    X_train_lle, X_test_lle, y_train_lle, y_test_lle = train_test_split(X_lle, y, test_size=0.3, random_state=gtid)

    times_train=[]
    times_test=[]
    # Train and evaluate on original data
    nn0 = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=gtid)
    begin_time=datetime.now()
    
    nn0.fit(X_train_orig, y_train_orig)
    end_time=datetime.now()
    y_pred_orig = nn0.predict(X_test_orig)
    end_time_2=datetime.now()
    times_train.append(end_time-begin_time)
    times_test.append(end_time_2-end_time)
    accuracy_orig = accuracy_score(y_test_orig, y_pred_orig)

    # Train and evaluate on PCA reduced data
    nn1 = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=gtid)
    begin_time=datetime.now()
    nn1.fit(X_train_pca, y_train_pca)
    end_time=datetime.now()
    y_pred_pca = nn1.predict(X_test_pca)
    end_time_2=datetime.now()
    times_train.append(end_time-begin_time)
    times_test.append(end_time_2-end_time)
    accuracy_pca = accuracy_score(y_test_pca, y_pred_pca)

    # Train and evaluate on ICA reduced data
    nn2 = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=gtid)
    begin_time=datetime.now()
    nn2.fit(X_train_ica, y_train_ica)
    end_time=datetime.now()
    y_pred_ica = nn2.predict(X_test_ica)
    end_time_2=datetime.now()
    times_train.append(end_time-begin_time)
    times_test.append(end_time_2-end_time)
    accuracy_ica = accuracy_score(y_test_ica, y_pred_ica)

    # Train and evaluate on RP reduced data
    nn3 = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=gtid)
    begin_time=datetime.now()
    nn3.fit(X_train_rp, y_train_rp)
    end_time=datetime.now()
    y_pred_rp = nn3.predict(X_test_rp)
    end_time_2=datetime.now()
    times_train.append(end_time-begin_time)
    times_test.append(end_time_2-end_time)
    accuracy_rp = accuracy_score(y_test_rp, y_pred_rp)

    # Train and evaluate on LLE reduced data
    nn4 = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=gtid)
    begin_time=datetime.now()
    nn4.fit(X_train_lle, y_train_lle)
    end_time=datetime.now()
    y_pred_lle = nn4.predict(X_test_lle)
    end_time_2=datetime.now()
    times_train.append(end_time-begin_time)
    times_test.append(end_time_2-end_time)
    accuracy_lle = accuracy_score(y_test_lle, y_pred_lle)



    # Print results
    print(f'Accuracy/Train/Test with Original: {accuracy_orig,str(times_train[0]),str(times_test[0])}')
    print(f'Accuracy/Train/Test with PCA: {accuracy_pca,str(times_train[1]),str(times_test[1])}')
    print(f'Accuracy/Train/Test with ICA: {accuracy_ica,str(times_train[2]),str(times_test[2])}')
    print(f'Accuracy/Train/Test with RP: {accuracy_rp,str(times_train[3]),str(times_test[3])}')
    print(f'Accuracy/Train/Test with LLE: {accuracy_lle,str(times_train[4]),str(times_test[4])}')

if __name__ == '__main__':
    main()