from abc import ABC, abstractmethod


class TextPrep(ABC):
    def fit(self, X, y=None):
        """Fits the `TextPrep` step. Considered a no-op."""
        return self

    def fit_partial(self, X, y=None):
        """Partially fits the `TextPrep` step. Considered a no-op."""
        return self

    @abstractmethod
    def encode_single(self, x):
        pass

    def pipe(self, X):
        for x in X:
            yield self.encode_single(x)

    def transform(self, X, y=None):
        """Transforms a batch of data."""
        return [self.encode_single(x) for x in X]
