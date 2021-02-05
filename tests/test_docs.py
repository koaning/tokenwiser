from tokenwiser.tok import SpacyTokenizer, WhiteSpaceTokenizer
from tokenwiser.prep import Cleaner, HyphenPrep, SpacyMorphPrep, SpacyPosPrep, SpacyLemmaPrep, YakePrep, TextConcat, make_concat

import pytest
from mktestdocs import check_docstring

components = [
    SpacyTokenizer, WhiteSpaceTokenizer, Cleaner, HyphenPrep, SpacyMorphPrep, SpacyPosPrep, SpacyLemmaPrep, YakePrep, TextConcat, make_concat
]


@pytest.mark.parametrize("obj", components, ids=lambda d: d.__qualname__)
def test_member(obj):
    check_docstring(obj)
