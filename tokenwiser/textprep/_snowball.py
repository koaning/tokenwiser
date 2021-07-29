import snowballstemmer
from sklearn.base import BaseEstimator

from ._prep import TextPrep




class SnowballTextPrep(TextPrep, BaseEstimator):
    """
    Applies the snowball stemmer to the text.

    There are 26 languages supported, for the full list check the list on the 
    lefthand side on [pypi](https://pypi.org/project/snowballstemmer/).

    Usage:

    ```python
    from tokenwiser.textprep import SnowballTextPrep

    single = SnowballTextPrep(language='english').encode_single("Dogs like running")
    assert single == "Dog like run"
    multi = Cleaner().transform(["Dogs like running", "Cats like sleeping"])
    assert multi == ["Dog like run", "Cat like sleep"]
    ```
    """

    def __init__(self, language='english'):
        self.stemmer = snowballstemmer.stemmer(language)

    def encode_single(self, x: str):
        return " ".join(self.stemmer.stemWords(x))
