from tokenwiser.tok._tok import Tok

from sklearn.base import BaseEstimator


class WhiteSpaceTokenizer(Tok, BaseEstimator):
    """
    A simple tokenizer that simple splits on whitespace.

    Usage:

    ```python
    from tokenwiser.tok import WhiteSpaceTokenizer

    single = WhiteSpaceTokenizer().encode_single("hello world")
    assert single == ["hello", "world"]
    multi = WhiteSpaceTokenizer().transform(["hello world", "it is me"])
    assert multi == [['hello', 'world'], ['it', 'is', 'me']]
    ```
    """
    def __init__(self):
        pass

    def encode_single(self, x):
        return [r for r in x.split(" ") if r != ""]
