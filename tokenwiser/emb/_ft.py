import numpy as np
from gensim.models import FastText
from tokenwiser.emb._emb import Emb


class Gensim(Emb):
    def __init__(self, dim=100, window=5, min_count=1, workers=4, epochs=10, min_n=3, max_n=6):
        self.dim = dim
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs

    def fit(self, X, y=None):
        self.model = FastText(size=self.dim, window=self.window, min_count=self.min_count, workers=self.workers, min_n=3, max_n=6)
        self.model.build_vocab(X)
        self.model.train(X, total_examples=self.model.corpus_count, epochs=10)
        return self

    def fit_partial(self, X):
        if not self.model:
            return self.fit(X)
        self.model.build_vocab(X, update=True)
        self.model.train(X, total_examples=self.model.corpus_count, epochs=self.epochs)
        return self

    @classmethod
    def from_file(cls):
        pass

    def transform(self, X, y=None):
        return np.array([self.encode_single(x) for x in X])

    def encode_single(self, x):
        if len(x) == 0:
            return np.zeros(self.dim)
        return np.array([self.model.wv[t] if (t in self.model.wv) else np.zeros(self.dim)
                         for t in x]).sum(axis=0)
