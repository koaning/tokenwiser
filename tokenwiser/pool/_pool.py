from abc import ABC, abstractmethod

import numpy as np


class Pool(ABC):
    def fit(self, X, y=None):
        return self

    def fit_partial(self, X):
        return self

    @abstractmethod
    def encode_single(self, emb):
        pass

    def pipe(self, X):
        for x in X:
            yield self.encode_single(x)

    def transform(self, embs, y=None):
        return np.array([self.encode_single(e) for e in embs])

    def save(self, folder):
        pass
