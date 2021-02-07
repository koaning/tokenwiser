from sklearn.pipeline import Pipeline, _name_estimators


class PartialPipeline(Pipeline):
    """
    Utility function to generate a `PartialPipeline`

    Arguments:
        steps: a collection of text-transformers

    ```python
    from tokenwiser.pipeline import PartialPipeline
    from tokenwiser.textprep import HyphenTextPrep, Cleaner

    tc = PartialPipeline([('clean', Cleaner()), ('hyp', HyphenTextPrep())])
    data = ["dinosaurhead", "another$$ sentence$$"]
    results = tc.fit_partial(data).transform(data)
    expected = ['di no saur head', 'an other  sen tence']

    assert results == expected
    ```
    """

    def __init__(self, steps):
        super().__init__(steps=steps)

    def fit_partial(self, X, y=None):
        """
        Fits the components, but allow for batches.
        """
        for name, step in self.steps:
            if not hasattr(step, "fit_partial"):
                raise ValueError(
                    f"Step {name} is a {step} which does not have `.fit_partial` implemented."
                )
        for name, step in self.steps:
            step.fit_partial(X, y)
        return self


def make_partial_pipeline(*steps):
    """
    Utility function to generate a `PartialPipeline`

    Arguments:
        steps: a collection of text-transformers

    ```python
    from tokenwiser.pipeline import make_partial_pipeline
    from tokenwiser.textprep import HyphenTextPrep, Cleaner

    tc = make_partial_pipeline(Cleaner(), HyphenTextPrep())
    data = ["dinosaurhead", "another$$ sentence$$"]
    results = tc.fit_partial(data).transform(data)
    expected = ['di no saur head', 'an other  sen tence']

    assert results == expected
    ```
    """
    return PartialPipeline(_name_estimators(steps))
