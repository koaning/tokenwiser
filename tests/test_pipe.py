import pathlib

from tokenwiser.tok import WhiteSpaceTokenizer
from tokenwiser.prep import HyphenPrep, Cleaner
from tokenwiser.pipe import Pipeline
from tokenwiser.pool import Pooling
from tokenwiser.emb import Gensim


def test_can_no_emb_pipeline():
    pipe = Pipeline(
        Cleaner(),
        WhiteSpaceTokenizer(),
    )
    assert pipe.encode_single("hello world!") == ["hello", "world"]


def test_can_train_pipeline():
    input_text = pathlib.Path("tests/data/nlp.txt").read_text().split("\n")
    pipe = Pipeline(
        Cleaner(),
        WhiteSpaceTokenizer(),
        Gensim(dim=25, epochs=1),
        Pooling()
    )
    assert pipe.fit(input_text).transform(input_text).shape == (len(input_text), 25)


def test_can_train_split_pipeline():
    input_text = pathlib.Path("tests/data/nlp.txt").read_text().split("\n")
    prep_pipe = Pipeline(
        Cleaner(),
        HyphenPrep(),
        WhiteSpaceTokenizer()
    )
    pipe = Pipeline(
        prep_pipe,
        Gensim(dim=25, epochs=1),
        Pooling()
    )
    assert pipe.fit(input_text).transform(input_text).shape == (len(input_text), 25)
