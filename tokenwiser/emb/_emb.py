from abc import ABC, abstractmethod


class Emb(ABC):
    def fit(self, X, y=None):
        raise NotImplementedError("not implemented")

    def fit_partial(self, X):
        raise NotImplementedError("not implemented")

    @abstractmethod
    def encode_single(self, x):
        pass

    def pipe(self, X, y=None):
        for x in X:
            yield self.encode_single(x)

    def transform(self, X, y=None):
        return [self.encode_single(x) for x in X]


class Embedding:
    def __init__(self, name, vec):
        self.name = name
        self.vec = vec
