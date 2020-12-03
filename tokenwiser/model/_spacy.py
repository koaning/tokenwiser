import spacy
from sklearn.base import BaseEstimator, ClassifierMixin


class SpacyClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, mod):
        if isinstance(mod, str):
            self.mod = spacy.load(mod)
        else:
            self.mod = mod

    def fit(self, X, y):
        pass

    def predict(self, X, y):
        pass
