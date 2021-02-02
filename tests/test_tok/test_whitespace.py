import pytest

from tokenwiser.tok import WhiteSpaceTokenizer


@pytest.mark.parametrize(
    "x_in,x_out",
    [("haleluja", ["haleluja"]), ("hello there world", ["hello", "there", "world"])],
)
def test_basic(x_in, x_out):
    assert WhiteSpaceTokenizer()(x_in) == x_out
