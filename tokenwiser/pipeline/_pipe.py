from sklearn.pipeline import Pipeline, _name_estimators


class PartialPipeline(Pipeline):
    """
    Utility function to generate a `PartialPipeline`

    Arguments:
        steps: a collection of text-transformers

    ```python
    from tokenwiser.pipe import make_partial_pipeline

    tc = make_partial_pipeline(HyphenPrep(), Cleaner())
    data = ["dinosaurhead", "another$$ sentence$$"]
    results = tc.fit_partial(data).transform(data)
    expected = ['di no saur head dinosaurhead', 'an other $$ sen tence$$ another sentence']

    assert results == expected
    ```
    """
    def __init__(self, steps):
        for name, step in steps:
            if not hasattr(step, "fit_partial"):
                raise ValueError(f"Step {name} is a {step} which does not have `.fit_partial` implemented.")
        super(Pipeline).__init__(steps=steps)

    def fit_partial(self, X, y):
        for name, step in self.steps:
            step.fit_partial(X, y)
        return self


def make_partial_pipeline(*steps):
    """
    Utility function to generate a `PartialPipeline`

    Arguments:
        steps: a collection of text-transformers

    ```python
    from tokenwiser.pipe import make_partial_pipeline

    tc = make_partial_pipeline(HyphenPrep(), Cleaner())
    data = ["dinosaurhead", "another$$ sentence$$"]
    results = tc.fit_partial(data).transform(data)
    expected = ['di no saur head dinosaurhead', 'an other $$ sen tence$$ another sentence']

    assert results == expected
    ```
    """
    return PartialPipeline(_name_estimators(steps))
