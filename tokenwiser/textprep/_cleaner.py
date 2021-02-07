from sklearn.base import BaseEstimator

from tokenwiser.prep._prep import Prep


class Cleaner(Prep, BaseEstimator):
    """
    Applies a lowercase and removes non-alphanum.

    Usage:

    ```python
    from tokenwiser.prep import Cleaner

    single = Cleaner().encode_single("$$$5 dollars")
    assert single == "5 dollars"
    multi = Cleaner().transform(["$$$5 dollars", "#hashtag!"])
    assert multi == ["5 dollars", "hashtag"]
    ```
    """
    def __init__(self):
        pass

    def encode_single(self, x: str):
        return "".join([c.lower() for c in x if c.isalnum() or c == " "])
