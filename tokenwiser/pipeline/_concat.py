from sklearn.pipeline import _name_estimators
from sklearn.base import BaseEstimator


class TextConcat(BaseEstimator):
    """
    A component like `FeatureUnion` but this also concatenates the text.

    Arguments:
        transformer_list: list of (name, text-transformer)-tuples

    Example:

    ```python
    from tokenwiser.textprep import HyphenTextPrep, Cleaner
    from tokenwiser.pipeline import TextConcat

    tc = TextConcat([("hyp", HyphenTextPrep()), ("clean", Cleaner())])
    results = tc.fit_transform(["dinosaurhead", "another$$ sentence$$"])
    expected = ['di no saur head dinosaurhead', 'an other $$ sen tence$$ another sentence']

    assert results == expected
    ```
    """
    def fit(self, X, y=None):
        """
        Fits the components in a single batch.
        """
        names = [n for n, t in self.transformer_list]
        if len(names) != len(set(names)):
            raise ValueError("Make sure that the names of each step are unique.")
        return self

    def partial_fit(self, X, y=None):
        """
        Fits the components, but allow for batches.
        """
        names = [n for n, t in self.transformer_list]
        if len(names) != len(set(names)):
            raise ValueError("Make sure that the names of each step are unique.")
        return self

    def transform(self, X, y=None):
        """
        Transformers the text.
        """
        names = [n for n, t in self.transformer_list]
        if len(names) != len(set(names)):
            raise ValueError("Make sure that the names of each step are unique.")
        results = {}
        for name, tfm in self.transformer_list:
            results[name] = tfm.transform(X)
        return [" ".join([results[n][i] for n in names]) for i in range(len(X))]

    def fit_transform(self, X, y=None):
        """
        Fits the components and transforms the text in one step.
        """
        return self.fit(X, y).transform(X, y)


def make_concat(*steps):
    """
    Utility function to generate a `TextConcat`

    Arguments:
        steps: a collection of text-transformers

    ```python
    from tokenwiser.textprep import HyphenTextPrep, Cleaner
    from tokenwiser.pipeline import make_concat

    tc = make_concat(HyphenTextPrep(), Cleaner())
    results = tc.fit_transform(["dinosaurhead", "another$$ sentence$$"])
    expected = ['di no saur head dinosaurhead', 'an other $$ sen tence$$ another sentence']

    assert results == expected
    ```
    """
    return TextConcat(_name_estimators(steps))
