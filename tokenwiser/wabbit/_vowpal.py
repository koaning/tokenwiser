import numpy as np
from vowpalwabbit import pyvw
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, ClassifierMixin


class VowpalWabbitClassifier(BaseEstimator, ClassifierMixin):
    """
    Vowpal Wabbit based text classifier.

    This object represents a simplified [Vowpal Wabbit](https://vowpalwabbit.org/) classifier that is
    compatible with scikit-learn. The only caveat is that the model expects
    text-arrays as opposed to numeric arrays.

    Arguments:
        n_loop: the number of times the fit step should apply to the training data
        n_gram: number of n_grams to encode as well
        learning_rate: the learning rate to apply while training

    Usage:

    ```python
    from tokenwiser.wabbit import VowpalWabbitClassifier

    clf = VowpalWabbitClassifier()

    X = [
        "this is friendly",
        "very friendly",
        "i do not like you",
        "the sky is blue"
    ]

    y = ["pos", "pos", "neg", "neutral"]

    # partial fitting
    for x_, y_ in zip(X, y):
        clf.partial_fit(x_, y_, classes=["pos", "neg", "neutral"])
    clf.predict(X)

    # batch fitting
    clf.fit(X, y).predict(X)
    ```
    """

    def __init__(self, n_loop: int = 1, n_gram: int = 1, learning_rate: float = 0.5):
        self.model = None
        self.n_loop = n_loop
        self.n_gram = n_gram
        self.learning_rate = learning_rate

    def fit(self, X, y):
        """
        Fit the model using X, y as training data.

        Arguments:
            X: array-like, shape=(n_columns, n_samples, ) training data, must be text.
            y: labels
        """
        return self.partial_fit(X, y, classes=list(set(y)))

    def partial_fit(self, X, y, classes):
        """
        Incremental fit on a batch of samples.

        Arguments:
            X: array-like, shape=(n_columns, n_samples, ) training data, must be text.
            y: labels
            classes: list of all the classes in the dataset
        """
        if not isinstance(X[0], str):
            raise ValueError("This model only accepts text as input.")
        if not self.model:
            self.classes_ = classes
            self.idx_to_cls_ = {i + 1: c for i, c in enumerate(self.classes_)}
            self.cls_to_idx_ = {c: i + 1 for i, c in enumerate(self.classes_)}
            self.model = pyvw.vw(
                quiet=True,
                oaa=len(classes),
                ngram=self.n_gram,
                learning_rate=self.learning_rate,
                loss_function="logistic",
                probabilities=True,
            )
        for loop in range(self.n_loop):
            for x_, y_ in zip(X, y):
                try:
                    self.model.learn(f"{self.cls_to_idx_[y_]} | {x_}")
                except RuntimeError as e:
                    ex = f"{self.cls_to_idx_[y_]} | {x_}"
                    raise RuntimeError(f"{e}\nculprit: {ex}")
        return self

    def predict_proba(self, X):
        """
        Return probability estimates for the test vector X.

        Arguments:
            X: array-like, shape=(n_columns, n_samples, ) training data, must be text.
        """
        check_is_fitted(self, ["classes_", "cls_to_idx_", "idx_to_cls_"])
        r = np.array([self.model.predict(f"| {x}") for x in X])
        return r / r.sum(axis=1).reshape(-1, 1)

    def predict(self, X):
        """
        Perform classification on an array of test vectors X.

        Arguments:
            X: array-like, shape=(n_columns, n_samples, ) training data, must be text.
        """
        argmax = self.predict_proba(X).argmax(axis=1)
        return np.array([self.idx_to_cls_[a + 1] for a in argmax])
