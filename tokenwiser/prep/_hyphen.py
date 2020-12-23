import pyphen
from sklearn.base import BaseEstimator

from tokenwiser.prep._prep import Prep


class HyphenPrep(Prep, BaseEstimator):
    """
    Hyphenate the text going in.

    Usage:

    ```python
    from tokenwiser.prep import HyphenPrep

    single = HyphenPrep().encode_single("dinosaurhead")
    assert single == "di no saur head"
    multi = HyphenPrep().transform(["geology", "astrology"])
    assert multi == ['geo logy', 'as tro logy']
    ```
    """
    def __init__(self, lang="en_GB"):
        self.dic = pyphen.Pyphen(lang=lang)

    def encode_single(self, x):
        return " ".join(self.dic.inserted(x).split("-", -1))
