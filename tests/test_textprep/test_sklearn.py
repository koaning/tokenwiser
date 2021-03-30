import pytest
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer

from tokenwiser.textprep import (
    Cleaner,
    HyphenTextPrep,
    SpacyMorphTextPrep,
    SpacyPosTextPrep,
    SpacyLemmaTextPrep,
    YakeTextPrep,
    PhoneticTextPrep,
    Identity,
    SentencePiecePrep
)
import spacy

nlp = spacy.load("en_core_web_sm")


prep_list = [
    Cleaner(),
    HyphenTextPrep(),
    PhoneticTextPrep(kind="soundex"),
    PhoneticTextPrep(kind="metaphone"),
    PhoneticTextPrep(kind="nysiis"),
    YakeTextPrep(),
    SpacyLemmaTextPrep(nlp),
    SpacyMorphTextPrep(nlp),
    SpacyPosTextPrep(nlp),
    Identity(),
    SentencePiecePrep(model_file="tests/data/en.model")
]


@pytest.mark.parametrize("prep", prep_list, ids=[str(d) for d in prep_list])
def test_pipeline_single(prep):
    X = ["hello world", "this is dog", "it should work"]
    pipe = Pipeline([("prep", prep), ("cv", CountVectorizer())])
    assert pipe.fit_transform(X).shape[0] == 3


@pytest.mark.parametrize("prep", prep_list, ids=[str(d) for d in prep_list])
def test_pipeline_single_clean_first(prep):
    X = ["hello world", "this is dog", "it should work"]
    pipe = Pipeline([("clean", Cleaner()), ("prep", prep), ("cv", CountVectorizer())])
    assert pipe.fit_transform(X).shape[0] == 3
