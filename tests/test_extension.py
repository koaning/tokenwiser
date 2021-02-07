import spacy
from tokenwiser.extension import attach_hyphen_extension


def test_hyphen_works():
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("this is a dinosaurhead")
    tok = doc[-1]
    attach_hyphen_extension()
    assert tok._.hyphen == ["di", "no", "saur", "head"]
