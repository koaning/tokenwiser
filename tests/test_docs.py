from tokenwiser.textprep import (
    Cleaner,
    HyphenTextPrep,
    SpacyMorphTextPrep,
    SpacyPosTextPrep,
    SpacyLemmaTextPrep,
    YakeTextPrep,
    PhoneticTextPrep,
)
from tokenwiser.pipeline import (
    TextConcat,
    PartialPipeline,
    make_partial_pipeline,
    make_concat,
)
from tokenwiser.extension import attach_hyphen_extension, attach_sklearn_extension, sklearn_method
from tokenwiser.component import attach_sklearn_categoriser

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
    attach_hyphen_extension,
    attach_sklearn_extension,
    sklearn_method,
    attach_sklearn_categoriser,
]


@pytest.mark.parametrize("obj", components, ids=lambda d: d.__qualname__)
def test_member(obj):
    check_docstring(obj)
