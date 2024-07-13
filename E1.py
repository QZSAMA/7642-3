from methods import * 


def main():
    # Random Seed
    gtid=903789757
    # Load datasets
    X_iris,y_iris,X_diabetes,y_diabetes = loan_datasets()

    

    
    # Number of clusters 
    # 3 for Iris mneans 3 different type of flower 
    # 2 for Diabetes, means whether got or not got diabetes)
    n_clusters_iris = 3
    n_clusters_diabetes = 2
    e1(X_iris,y_iris,n_clusters_iris,"Iris",gtid)
    e1(X_diabetes,y_diabetes,n_clusters_diabetes,"Diabetes",gtid)


if __name__ == '__main__':
    main()