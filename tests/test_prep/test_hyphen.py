import pytest

from tokenwiser.prep import HyphenPrep


@pytest.mark.parametrize("x_in,x_out", [("haleluja", "hale lu ja"), ("hello", "hello")])
def test_basic(x_in, x_out):
    assert HyphenPrep().encode_single(x_in) == x_out
