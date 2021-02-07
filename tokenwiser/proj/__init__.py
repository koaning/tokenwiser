import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted


class BinaryRandomProjection(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=100, random_seed=42, threshold=0.0):
        self.n_components = n_components
        self.random_seed = random_seed
        self.threshold = threshold

    def fit(self, X, y=None):
        X = check_array(X)
        np.random.seed(self.random_seed)
        self.proj_ = np.random.normal(0, 1, (X.shape[1], self.n_components))
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ["proj_"])
        return (X @ self.proj_ > self.threshold).astype(np.int8)


def proj_away(x, y):
    """project y away from x"""
    return x.dot(x) / y.dot(y) * x


def select_random_rows(X):
    i1, i2 = np.random.randint(0, X.shape[0], 2)
    return X[i1, :], X[i2, :]


class PointSplitProjection(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=100, random_seed=42):
        self.n_components = n_components
        self.random_seed = random_seed

    def fit(self, X, y=None):
        X = check_array(X)
        self.X_ = X
        self.indices_ = [
            tuple(np.random.randint(0, X.shape[0], 2)) for t in range(self.n_components)
        ]
        return self

    def generate_feature_(self, new_X, i):
        i1, i2 = self.indices_[i]
        v1, v2 = self.X_[i1, :], self.X_[i2, :]
        m = np.array([v1, v2]).mean(axis=0)
        return new_X @ (proj_away(v2 - v1, m)) > m.dot(proj_away(v2 - v1, m))

    def transform(self, X, y=None):
        check_is_fitted(self, ["X_", "indices_"])
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError(
                f"shapes train/transform do not match. {X.shape[1]} vs {self.X_.shape[1]}"
            )
        result = np.zeros((X.shape[0], self.n_components))
        for col in range(self.n_components):
            result[:, col] = self.generate_feature_(X, col)
        return result
