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
    results = tc.partial_fit(data).transform(data)
    expected = ['di no saur head', 'an other  sen tence']

    assert results == expected
    ```
    """
    def partial_fit(self, X, y=None, classes=None, **kwargs):
        """
        Fits the components, but allow for batches.
        """
        for name, step in self.steps:
            if not hasattr(step, "partial_fit"):
                raise ValueError(
                    f"Step {name} is a {step} which does not have `.partial_fit` implemented."
                )
        for name, step in self.steps:
            if hasattr(step, "predict"):
                step.partial_fit(X, y, classes=classes, **kwargs)
            else:
                step.partial_fit(X, y)
            if hasattr(step, "transform"):
                X = step.transform(X)
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
    results = tc.partial_fit(data).transform(data)
    expected = ['di no saur head', 'an other  sen tence']

    assert results == expected
    ```
    """
    return PartialPipeline(_name_estimators(steps))
