from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def preprocess_data(X, y):

    # Normalize data
    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    # Reduce dimensions for quantum compatibility
    pca = PCA(n_components=2)

    X = pca.fit_transform(X)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test, scaler, pca