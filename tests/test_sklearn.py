import itertools as it

import pytest
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer

from tokenwiser.prep import Cleaner, HyphenPrep, PhoneticPrep

prep_list = [
    Cleaner(),
    HyphenPrep(),
    PhoneticPrep(kind='soundex'),
    PhoneticPrep(kind='metaphone'),
    PhoneticPrep(kind='nysiis')
]


@pytest.mark.parametrize('prep', prep_list)
def test_pipeline_single(prep):
    X = ['hello world', 'this is dog', 'it should work']
    pipe = Pipeline([
        ('prep', prep),
        ('cv', CountVectorizer())
    ])
    assert pipe.fit_transform(X).shape[0] == 3


@pytest.mark.parametrize('prep', prep_list)
def test_pipeline_single_clean_first(prep):
    X = ['hello world', 'this is dog', 'it should work']
    pipe = Pipeline([
        ('clean', Cleaner()),
        ('prep', prep),
        ('cv', CountVectorizer())
    ])
    assert pipe.fit_transform(X).shape[0] == 3
