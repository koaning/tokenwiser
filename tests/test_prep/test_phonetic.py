import pytest

from tokenwiser.prep import PhoneticPrep


@pytest.mark.parametrize("x_in,x_out", [("haleluja", "H442"), ("hello there world", "H400 T600 W643")])
def test_basic(x_in, x_out):
    assert PhoneticPrep().encode_single(x_in) == x_out
