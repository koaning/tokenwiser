from sklearn.base import BaseEstimator

from tokenwiser.prep._prep import Prep


class Cleaner(Prep, BaseEstimator):
    """
    Applies a lowercase and removes non-alphanum.
    """
    def __init__(self):
        pass

    def encode_single(self, x: str):
        return "".join([c.lower() for c in x if c.isalnum() or c == " "])
