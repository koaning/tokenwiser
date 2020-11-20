import pytest

from tokenwiser.tok import WhiteSpaceTokenizer


@pytest.mark.parametrize("x_in,x_out", [("haleluja", ["haleluja"]), ("hello there world", ["hello", "there", "world"])])
def test_basic(x_in, x_out):
    assert WhiteSpaceTokenizer().encode_single(x_in) == x_out


def test_transform():
    X = ["hello there", "this is a test", "another one"]
    result = WhiteSpaceTokenizer().transform(X)
    assert len(result[0]) == 2
    assert len(result[1]) == 4
    assert len(result[2]) == 2
