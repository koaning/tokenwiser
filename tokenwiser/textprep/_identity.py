from sklearn.base import BaseEstimator
from ._prep import TextPrep


class Identity(TextPrep, BaseEstimator):
    """
    Keeps the text as is. Can be used as a placeholder in a pipeline.

    Usage:

    ```python
    from tokenwiser.textprep import Identity

    text = ["hello", "world"]
    example = Identity().transform(text)

    assert example == ["hello", "world"]
    ```

    The main use-case is as a placeholder.

    ```
    from tokenwiser.pipeline import make_concat
    from sklearn.pipeline import make_pipeline, make_union

    from tokenwiser.textprep import Cleaner, Identity, HyphenTextPrep

    pipe = make_pipeline(
        Cleaner(),
        make_concat(Identity(), HyphenTextPrep()),
    )
    ```
    """

    def __init__(self):
        pass

    def encode_single(self, x):
        return x

    def transform(self, X, y=None):
        return X

    def fit(self, X, y=None):
        return self

    def partial_fit(self, X, y=None):
        return self
