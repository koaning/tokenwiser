import pytest
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer

from tokenwiser.prep import Cleaner, HyphenPrep, PhoneticPrep
from tokenwiser.pool import Pooling
from tokenwiser.tok import WhiteSpaceTokenizer
from tokenwiser.emb import Gensim

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


@pytest.mark.parametrize('prep', prep_list)
def test_long_pipeline(prep):
    X = ['hello world', 'this is dog', 'it should work']
    pipe = Pipeline([
        ('clean', Cleaner()),
        ('prep', prep),
        ('cv', WhiteSpaceTokenizer()),
        ('emb', Gensim(dim=25, epochs=1)),
        ('pool', Pooling())
    ])
    assert pipe.fit_transform(X).shape == (3, 25)
