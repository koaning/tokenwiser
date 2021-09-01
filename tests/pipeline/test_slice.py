from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression

from tokenwiser.textprep import Cleaner, Identity, HyphenTextPrep
from tokenwiser.pipeline import PartialPipeline, make_partial_pipeline, make_concat


def test_can_slice_pipeline():
    """If we slice a pipeline, we should get a new pipeline object"""
    pipe1 = make_partial_pipeline(
        Cleaner(),
        make_concat(
            Identity(),
            HyphenTextPrep(),
        ),
        HashingVectorizer(),
        LogisticRegression()
    )

    slice = pipe1[:-1]
    assert isinstance(slice, PartialPipeline)
