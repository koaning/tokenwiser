import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from tokenwiser.textprep import Cleaner, Identity, HyphenTextPrep
from tokenwiser.pipeline import PartialPipeline, PartialFeatureUnion


def test_shape_doubles():
    """If we concatenate using a partial union. It should increase in size."""
    pipe1 = PartialPipeline(
        [
            ("clean", Cleaner()),
            (
                "union",
                PartialFeatureUnion(
                    [
                        (
                            "full_text_pipe",
                            PartialPipeline(
                                [
                                    ("identity", Identity()),
                                    ("hash1", HashingVectorizer()),
                                ]
                            ),
                        )
                    ]
                ),
            ),
        ]
    )

    pipe2 = PartialPipeline(
        [
            ("clean", Cleaner()),
            (
                "union",
                PartialFeatureUnion(
                    [
                        (
                            "full_text_pipe",
                            PartialPipeline(
                                [
                                    ("identity", Identity()),
                                    ("hash1", HashingVectorizer()),
                                ]
                            ),
                        ),
                        (
                            "hyphen_pipe",
                            PartialPipeline(
                                [
                                    ("hyphen", HyphenTextPrep()),
                                    ("hash2", HashingVectorizer()),
                                ]
                            ),
                        ),
                    ]
                ),
            ),
        ]
    )

    X = [
        "i really like this post",
        "thanks for that comment",
        "i enjoy this friendly forum",
        "this is a bad post",
        "i dislike this article",
        "this is not well written",
    ]

    y = np.array([1, 1, 1, 0, 0, 0])

    p1 = pipe1.partial_fit(X, y, classes=[0, 1]).transform(X)
    p2 = pipe2.partial_fit(X, y, classes=[0, 1]).transform(X)

    assert p1.shape[1] * 2 == p2.shape[1]
