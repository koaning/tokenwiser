import pytest

from tokenwiser.textprep import HyphenTextPrep


@pytest.mark.parametrize("x_in,x_out", [("haleluja", "hale lu ja"), ("hello", "hello")])
def test_basic(x_in, x_out):
    assert HyphenTextPrep().encode_single(x_in) == x_out
