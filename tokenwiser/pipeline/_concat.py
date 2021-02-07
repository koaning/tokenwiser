from sklearn.pipeline import _name_estimators
from sklearn.base import BaseEstimator


class TextConcat(BaseEstimator):
    """
    A component like `FeatureUnion` but this also concatenates the text.

    Arguments:
        transformer_list: list of (name, text-transformer)-tuples

    Example:

    ```python
    from tokenwiser.prep import HyphenPrep, Cleaner, TextConcat

    tc = TextConcat([("hyp", HyphenPrep()), ("clean", Cleaner())])
    results = tc.fit_transform(["dinosaurhead", "another$$ sentence$$"])
    expected = ['di no saur head dinosaurhead', 'an other $$ sen tence$$ another sentence']

    assert results == expected
    ```
    """
    def __init__(self, transformer_list):
        self.transformer_list = transformer_list

    def fit(self, X, y=None):
        """
        Fits the components.

        Arguments:
            X: list of text, to be transformer
            y: a label, will be handled by the `Pipeline`-API
        """
        names = [n for n, t in self.transformer_list]
        if len(names) != len(set(names)):
            raise ValueError(f"Make sure that the names of each step are unique.")
        return self

    def fit_partial(self, X, y=None):
        """
        Fits the components.

        Arguments:
            X: list of text, to be transformer
            y: a label, will be handled by the `Pipeline`-API
        """
        names = [n for n, t in self.transformer_list]
        if len(names) != len(set(names)):
            raise ValueError(f"Make sure that the names of each step are unique.")
        return self

    def transform(self, X, y=None):
        """
        Transformers the text.

        Arguments:
            X: list of text, to be transformer
            y: a label, will be handled by the `Pipeline`-API
        """
        names = [n for n, t in self.transformer_list]
        if len(names) != len(set(names)):
            raise ValueError(f"Make sure that the names of each step are unique.")
        results = {}
        for name, tfm in self.transformer_list:
            results[name] = tfm.transform(X)
        return [" ".join([results[n][i] for n in names]) for i in range(len(X))]

    def fit_transform(self, X, y=None):
        """
        Fits the components and transforms the text in one step.

        Arguments:
            X: list of text, to be transformer
            y: a label, will be handled by the `Pipeline`-API
        """
        return self.fit(X, y).transform(X, y)


def make_concat(*steps):
    """
    Utility function to generate a `TextConcat`

    Arguments:
        steps: a collection of text-transformers

    ```python
    from tokenwiser.prep import HyphenPrep, Cleaner, make_concat

    tc = make_concat(HyphenPrep(), Cleaner())
    results = tc.fit_transform(["dinosaurhead", "another$$ sentence$$"])
    expected = ['di no saur head dinosaurhead', 'an other $$ sen tence$$ another sentence']

    assert results == expected
    ```
    """
    return TextConcat(_name_estimators(steps))
