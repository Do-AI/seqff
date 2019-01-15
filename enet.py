import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    # Load and split data
    # X = np.load('files/X.npy')
    # y = np.load('files/y.npy')
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X_train = np.load('files/X_train.npy')
    X_test = np.load('files/X_test.npy')
    y_train = np.load('files/y_train.npy')
    y_test = np.load('files/y_test.npy')

    # Preprocessing - standardization
    st_scaler = StandardScaler()
    X_train_st = st_scaler.fit_transform(X_train)
    X_test_st = st_scaler.fit_transform(X_test)

    # Make model
    elasticnet_st = ElasticNetCV(cv=5, verbose=True)
    elasticnet_st.fit(X_train_st, y_train)

    # Evaluation model
    elasticnet_st.score(X_test_st, y_test)


if __name__ == '__main__':
    main()