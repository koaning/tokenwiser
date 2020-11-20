from sklearn.feature_extraction.text import CountVectorizer


class SlidingWindowTokenizer:
    def __init__(self, ngram_range=(3, 3), analyzer="char_wb"):
        self.ngram_range = ngram_range
        self.analyzer = analyzer

    def fit(self, X, y=None):
        self.cv_ = CountVectorizer(ngram_range=self.ngram_range, analyzer="char_wb").fit(X)
        self.encoder = self.cv_.build_analyzer()
        return self

    def encode(self, x):
        return list(set([t.replace(" ", "") for t in self.encoder(x)]))

    def transform(self, X, y=None):
        return [self.encode(x) for x in X]
