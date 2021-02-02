import jellyfish
from sklearn.base import BaseEstimator

from tokenwiser.prep._prep import Prep


class PhoneticPrep(Prep, BaseEstimator):
    """
    The ProneticPrep object prepares strings by encoding them phonetically.

    Arguments:
        kind: type of encoding, either `"soundex"`, "`metaphone`" or `"nysiis"`

    Usage:

    ```python
    from tokenwiser.prep import PhoneticPrep

    nlp = spacy.load("en_core_web_sm")
    example1 = PhoneticPrep(kind="soundex").encode_single("dinosaurus book")
    example2 = PhoneticPrep(kind="metaphone").encode_single("dinosaurus book")
    example3 = PhoneticPrep(kind="nysiis").encode_single("dinosaurus book")

    assert example1 == 'D526 B200'
    assert example2 == 'TNSRS BK'
    assert example3 == 'DANASAR BAC'
    ```
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
