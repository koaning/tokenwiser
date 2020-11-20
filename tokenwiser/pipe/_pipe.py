class Pipeline:
    def __init__(self, *args):
        self.pipeline = list(args)

    def fit(self, X):
        result = X
        for pipe in self.pipeline:
            result = pipe.fit(result).transform(result)
        return self

    def transform(self, X):
        result = X
        for pipe in self.pipeline:
            result = pipe.transform(result)
        return result

    def encode_single(self, x):
        result = x
        for pipe in self.pipeline:
            result = pipe.encode_single(result)
        return result
