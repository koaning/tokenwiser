import numpy as np
from tokenwiser.wabbit import VowpalWabbitClassifier

X = [
    "i really like this post",
    "thanks for that comment",
    "i enjoy this friendly forum",
    "this is a bad post",
    "i dislike this article",
    "this is not well written"
]

y = np.array([1, 1, 1, 0, 0, 0])


def test_wabbit_shape_sensible():
    assert VowpalWabbitClassifier().fit(X, y).predict(X).shape[0] == 6
    assert VowpalWabbitClassifier().fit(X, y).predict_proba(X).shape == (6, 2)
