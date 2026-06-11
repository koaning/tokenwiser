"""Type-aware parsing of the public ``train`` search-space arguments.

These exercise validation only and never reach the trainer, so they are fast.
Parsing raises ``ValueError`` *before* the Rust core is invoked, so a bad
argument fails even though ``path`` does not exist.
"""

import pytest

import tokenwiser

NOPATH = "/nonexistent/does-not-exist.jsonl"


def _train(**kwargs):
    return tokenwiser.train(NOPATH, **kwargs)


# --- scalar / list / tuple semantics ------------------------------------- #
def test_scalar_string_ok_but_unknown_value_rejected():
    with pytest.raises(ValueError, match="tokenizer"):
        _train(tokenizer="klingon")


def test_string_choice_validates_each_member():
    with pytest.raises(ValueError, match="feature_value"):
        _train(feature_value=["binary", "nonsense"])


def test_ngram_fixed_tuple_vs_choice_list():
    # A bad fixed tuple is rejected for its own reason (min > max)...
    with pytest.raises(ValueError, match="ngram_range"):
        _train(ngram_range=(2, 1))
    # ...and a malformed choice member too.
    with pytest.raises(ValueError, match="ngram_range"):
        _train(ngram_range=[(1, 1), (0, 2)])


def test_int_range_is_inclusive_tuple_only():
    with pytest.raises(ValueError, match="epochs"):
        _train(epochs=(10, 2))  # lo > hi
    with pytest.raises(ValueError, match="n_hashes"):
        _train(n_hashes=(1, 2, 3))  # not a 2-tuple


def test_log_range_rejects_zero_and_negative():
    with pytest.raises(ValueError, match="log-uniform"):
        _train(learning_rate=(0.0, 0.1))
    with pytest.raises(ValueError, match="l2"):
        _train(l2=(-1e-3, 1e-2))


def test_l1_zero_is_a_valid_fixed_scalar():
    # 0.0 must be allowed as a *fixed* value (it just cannot be a log range).
    # Parsing passes; the call then fails only because the file is missing.
    with pytest.raises(ValueError, match="cannot open"):
        _train(l1=0.0)


def test_batch_size_is_fixed_execution_setting():
    with pytest.raises(ValueError, match="batch_size"):
        _train(batch_size=(32, 1024))


def test_cv_must_be_at_least_two():
    with pytest.raises(ValueError, match="cv"):
        _train(cv=1)


def test_bool_choice_for_lowercase():
    with pytest.raises(ValueError, match="lowercase"):
        _train(lowercase=["yes", "no"])


def test_numeric_list_choice_is_accepted():
    # Discrete non-contiguous numeric choices parse fine; only the missing
    # file stops the call.
    with pytest.raises(ValueError, match="cannot open"):
        _train(n_hashes=[1, 4, 8], n_features=[1024, 4096])
