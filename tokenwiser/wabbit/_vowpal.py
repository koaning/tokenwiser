import numpy as np
from vowpalwabbit import pyvw
from sklearn.utils.validation import check_is_fitted


class VowpalWabbitClassifier:
    def __init__(self, n_loop=1, n_gram=1, learning_rate=0.5, quadratic=False):
        self.model = None
        self.n_loop = n_loop
        self.n_gram = n_gram
        self.learning_rate = learning_rate
        self.quadratic = quadratic

    def fit(self, X, y):
        return self.fit_partial(X, y, classes=list(set(y)))

    def fit_partial(self, X, y, classes):
        if not isinstance(X[0], str):
            raise ValueError("This model only accepts text as input.")
        if not self.model:
            self.classes_ = classes
            self.idx_to_cls_ = {i + 1: c for i, c in enumerate(self.classes_)}
            self.cls_to_idx_ = {c: i + 1 for i, c in enumerate(self.classes_)}
            self.model = pyvw.vw(quiet=True, oaa=len(classes), ngram=self.n_gram,
                                 learning_rate=self.learning_rate, quadratic=self.quadratic,
                                 loss_function='logistic', probabilities=True)
        for loop in range(self.n_loop):
            for x_, y_ in zip(X, y):
                self.model.learn(f"{self.cls_to_idx_[y_]} | {x_}")
        return self

    def predict_proba(self, X):
        check_is_fitted(self, ["classes_", "cls_to_idx_", "idx_to_cls_"])
        r = np.array([self.model.predict(f"| {x}") for x in X])
        return r / r.sum(axis=1).reshape(-1, 1)

    def predict(self, X):
        argmax = self.predict_proba(X).argmax(axis=1)
        return np.array([self.idx_to_cls_[a + 1] for a in argmax])
