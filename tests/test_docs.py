from tokenwiser.textprep import (
    Cleaner,
    HyphenTextPrep,
    SpacyMorphTextPrep,
    SpacyPosTextPrep,
    SpacyLemmaTextPrep,
    YakeTextPrep,
    PhoneticTextPrep
)
from tokenwiser.pipeline import (
    TextConcat,
    PartialPipeline,
    make_partial_pipeline,
    make_concat,
)

import pytest
from mktestdocs import check_docstring

components = [
    Cleaner,
    HyphenTextPrep,
    SpacyMorphTextPrep,
    SpacyPosTextPrep,
    SpacyLemmaTextPrep,
    PhoneticTextPrep,
    YakeTextPrep,
    TextConcat,
    PartialPipeline,
    make_partial_pipeline,
    make_concat,
]


@pytest.mark.parametrize("obj", components, ids=lambda d: d.__qualname__)
def test_member(obj):
    check_docstring(obj)
