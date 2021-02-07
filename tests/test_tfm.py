import pytest

from tokenwiser.common import flatten
from tokenwiser.proj import BinaryRandomProjection, PointSplitProjection

from tests.conftest import (
    nonmeta_checks,
    general_checks,
    transformer_checks,
    select_tests,
)


@pytest.mark.parametrize(
    "test_fn",
    select_tests(
        flatten([nonmeta_checks, transformer_checks, general_checks]),
        exclude=[
            "check_transformer_data_not_an_array",
            "check_estimators_nan_inf",
            "check_fit2d_predict1d",
            "check_sample_weights_invariance",
        ],
    ),
)
def test_estimator_checks_binary(test_fn):
    random_proj = BinaryRandomProjection(random_seed=42)
    test_fn(random_proj, random_proj)


@pytest.mark.parametrize(
    "test_fn",
    select_tests(
        flatten([nonmeta_checks, transformer_checks, general_checks]),
        exclude=[
            "check_transformer_data_not_an_array",
            "check_sample_weights_invariance",
            "check_estimators_nan_inf",
            "check_fit2d_predict1d",
            "check_transformer_general",
            "check_pipeline_consistency",
        ],
    ),
)
def test_estimator_checks_split(test_fn):
    random_proj = PointSplitProjection(random_seed=42)
    test_fn(random_proj, random_proj)
