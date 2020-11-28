import jellyfish
from sklearn.base import BaseEstimator

from tokenwiser.prep._prep import Prep


class PhoneticPrep(Prep, BaseEstimator):
    """
    The ProneticPrep object prepares strings by encoding them phonetically.

    Arguments:
        kind: type of encoding, either `"soundex"`, "`metaphone`" or `"nysiis"`
    """

    def __init__(self, kind="soundex"):
        methods = {
            "soundex": jellyfish.soundex,
            "metaphone": jellyfish.metaphone,
            "nysiis": jellyfish.nysiis,
        }
        self.method = methods[kind]

    def encode_single(self, x):
        return " ".join([self.method(d) for d in x.split(" ")])
