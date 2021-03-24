import numpy as np
from tokenwiser.wabbit import VowpalWabbitClassifier
from tokenwiser.pipeline import make_partial_pipeline
from tokenwiser.textprep import Cleaner

X = [
    "i really like this post",
    "thanks for that comment",
    "i enjoy this friendly forum",
    "this is a bad post",
    "i dislike this article",
    "this is not well written"
]

y = np.array([1, 1, 1, 0, 0, 0])


def test_wabbit_fit_shape_sensible():
    assert VowpalWabbitClassifier().fit(X, y).predict(X).shape[0] == 6
    assert VowpalWabbitClassifier().fit(X, y).predict_proba(X).shape == (6, 2)


def test_wabbit_pipeline():
    pipe = make_partial_pipeline(Cleaner(),
                                 VowpalWabbitClassifier(n_loop=1, n_gram=1, learning_rate=0.1))
    for i in range(5):
        pipe.partial_fit(X, y, classes=list(set(y)))
