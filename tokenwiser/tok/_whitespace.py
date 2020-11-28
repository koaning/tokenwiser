from tokenwiser.tok._tok import Tok

from sklearn.base import BaseEstimator


class WhiteSpaceTokenizer(Tok, BaseEstimator):
    def __init__(self):
        pass

    def encode_single(self, x):
        return [r for r in x.split(" ") if r != ""]
