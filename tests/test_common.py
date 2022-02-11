import pytest
import pathlib
from tokenwiser.common import load_coefficients, save_coefficients

import numpy as np
from sklearn.linear_model import SGDClassifier, LogisticRegression, PassiveAggressiveClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import make_pipeline
from tokenwiser.textprep import Cleaner, Identity, HyphenTextPrep
from tokenwiser.pipeline import make_partial_pipeline, make_partial_union


@pytest.mark.parametrize("clf_train", [LogisticRegression, SGDClassifier, PassiveAggressiveClassifier])
@pytest.mark.parametrize("clf_target", [LogisticRegression, SGDClassifier, PassiveAggressiveClassifier])
def test_load_save(clf_train, clf_target, tmpdir):
    """
    Ensure that we can save/load vectors.
    """
    clf = clf_train()
    pipe = make_pipeline(
        Cleaner(),
        make_partial_union(
            make_partial_pipeline(Identity(), HashingVectorizer()),
            make_partial_pipeline(HyphenTextPrep(), HashingVectorizer())
        ),
        clf
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

    pipe.fit(X, y)

    assert np.all(pipe.predict(X) == y)
    
    # Here we create in the new pipeline. 
    clf_new = clf_target()
    pipe_new = make_partial_pipeline(
        Cleaner(),
        make_partial_union(
            make_partial_pipeline(Identity(), HashingVectorizer()),
            make_partial_pipeline(HyphenTextPrep(), HashingVectorizer())
        ),
        clf
    )
    path = pathlib.Path(tmpdir, "coefs.h5")
    save_coefficients(clf, path)
    load_coefficients(clf_new, path)
    assert np.all(clf.intercept_ == clf_new.intercept_)
    assert np.all(clf.coef_ == clf_new.coef_)
    assert np.all(clf.classes_ == clf_new.classes_)
    assert np.all(pipe_new.predict(X) == y)
