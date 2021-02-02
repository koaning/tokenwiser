from tokenwiser.tok._tok import Tok

from sklearn.base import BaseEstimator


class WhiteSpaceTokenizer(Tok, BaseEstimator):
    """
    A simple tokenizer that simple splits on whitespace.

    Usage:

    ```python
    from tokenwiser.tok import WhiteSpaceTokenizer

    tok = WhiteSpaceTokenizer()
    single = tok("hello world")
    assert single == ["hello", "world"]
    ```
    """
    def __init__(self):
        pass

    def __call__(self, text):
        return [r for r in text.split(" ") if r != ""]
