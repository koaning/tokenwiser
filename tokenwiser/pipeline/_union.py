from sklearn.pipeline import FeatureUnion, _name_estimators


class PartialFeatureUnion(FeatureUnion):
    """
    A `PartialFeatureUnion` is a `FeatureUnion` but able to `.partial_fit`.

    Arguments:
        transformer_list: a list of transformers to apply and concatenate

    Example:

    ```python
    import numpy as np
    from sklearn.linear_model import SGDClassifier
    from sklearn.feature_extraction.text import HashingVectorizer

    from tokenwiser.textprep import Cleaner, Identity, HyphenTextPrep
    from tokenwiser.pipeline import PartialPipeline, PartialFeatureUnion

    pipe = PartialPipeline([
        ("clean", Cleaner()),
        ("union", PartialFeatureUnion([
            ("full_text_pipe", PartialPipeline([
                ("identity", Identity()),
                ("hash1", HashingVectorizer()),
            ])),
            ("hyphen_pipe", PartialPipeline([
                ("hyphen", HyphenTextPrep()),
                ("hash2", HashingVectorizer()),
            ]))
        ])),
        ("clf", SGDClassifier())
    ])

    X = [
        "i really like this post",
        "thanks for that comment",
        "i enjoy this friendly forum",
        "this is a bad post",
        "i dislike this article",
        "this is not well written"
    ]

    y = np.array([1, 1, 1, 0, 0, 0])

    for loop in range(3):
        pipe.partial_fit(X, y, classes=[0, 1])

    assert np.all(pipe.predict(X) == np.array([1, 1, 1, 0, 0, 0]))
    ```
    """

    def __init__(self, transformer_list):
        super().__init__(transformer_list=transformer_list)

    def partial_fit(self, X, y=None, classes=None, **kwargs):
        """
        Fits the components, but allow for batches.
        """
        for name, step in self.transformer_list:
            if not hasattr(step, "partial_fit"):
                raise ValueError(
                    f"Step {name} is a {step} which does not have `.partial_fit` implemented."
                )
        for name, step in self.transformer_list:
            if hasattr(step, "predict"):
                step.partial_fit(X, y, classes=classes, **kwargs)
            else:
                step.partial_fit(X, y)
        return self


def make_partial_union(*transformer_list):
    """
    Utility function to generate a `PartialFeatureUnion`

    Arguments:
        transformer_list: a list of transformers to apply and concatenate

    Example:

    ```python
    import numpy as np
    from sklearn.linear_model import SGDClassifier
    from sklearn.feature_extraction.text import HashingVectorizer

    from tokenwiser.textprep import Cleaner, Identity, HyphenTextPrep
    from tokenwiser.pipeline import make_partial_pipeline, make_partial_union

    pipe = make_partial_pipeline(
        Cleaner(),
        make_partial_union(
            make_partial_pipeline(Identity(), HashingVectorizer()),
            make_partial_pipeline(HyphenTextPrep(), HashingVectorizer())
        ),
        SGDClassifier()
    )

    X = [
        "i really like this post",
        "thanks for that comment",
        "i enjoy this friendly forum",
        "this is a bad post",
        "i dislike this article",
        "this is not well written"
    ]

    y = np.array([1, 1, 1, 0, 0, 0])

    for loop in range(3):
        pipe.partial_fit(X, y, classes=[0, 1])

    assert np.all(pipe.predict(X) == np.array([1, 1, 1, 0, 0, 0]))
    ```
    """
    return PartialFeatureUnion(_name_estimators(transformer_list))
