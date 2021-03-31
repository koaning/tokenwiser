import pathlib
from tokenwiser.textprep import (
    Cleaner,
    Identity,
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
    PartialFeatureUnion,
    make_partial_pipeline,
    make_concat,
    make_partial_union,
)
from tokenwiser.extension import (
    attach_hyphen_extension,
    attach_sklearn_extension,
    sklearn_method,
)
from tokenwiser.component import attach_sklearn_categoriser

import pytest
from mktestdocs import check_docstring, check_md_file

components = [
    Cleaner,
    Identity,
    HyphenTextPrep,
    SpacyMorphTextPrep,
    SpacyPosTextPrep,
    SpacyLemmaTextPrep,
    PhoneticTextPrep,
    YakeTextPrep,
    TextConcat,
    PartialPipeline,
    PartialFeatureUnion,
    make_partial_pipeline,
    make_concat,
    make_partial_union,
    attach_hyphen_extension,
    attach_sklearn_extension,
    sklearn_method,
    attach_sklearn_categoriser,
]


@pytest.mark.parametrize("obj", components, ids=lambda d: d.__qualname__)
def test_member(obj):
    check_docstring(obj)


@pytest.mark.parametrize(
    "fpath", [str(p) for p in pathlib.Path("docs").glob("**/*.md")]
)
def test_fpath(fpath):
    check_md_file(fpath)
