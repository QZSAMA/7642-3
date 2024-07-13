from methods import * 





def main():
    # Random Seed
    gtid=903789757
    # Load datasets
    X_iris,y_iris,X_diabetes,y_diabetes = loan_datasets()

    # Number of components (for iris, let's keep 2, and for diabetes, let's keep 5)
    n_components_iris = 2
    n_components_diabetes = 5
    e2(X_iris,y_iris,n_components_iris,"Iris",gtid)
    e2(X_diabetes,y_diabetes,n_components_diabetes,"Diabetes",gtid)
    
    return 

if __name__ == '__main__':
    main()