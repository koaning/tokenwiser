import pytest

from tokenwiser.textprep import PhoneticTextPrep


@pytest.mark.parametrize(
    "x_in,x_out", [("haleluja", "H442"), ("hello there world", "H400 T600 W643")]
)
def test_soundex(x_in, x_out):
    assert PhoneticTextPrep(kind="soundex").encode_single(x_in) == x_out


@pytest.mark.parametrize(
    "x_in,x_out", [("haleluja", "HLLJ"), ("hello there world", "HL 0R WRLT")]
)
def test_metaphone(x_in, x_out):
    assert PhoneticTextPrep(kind="metaphone").encode_single(x_in) == x_out


@pytest.mark.parametrize(
    "x_in,x_out", [("haleluja", "HALALAJ"), ("hello there world", "HAL TAR WARLD")]
)
def test_nysiis(x_in, x_out):
    assert PhoneticTextPrep(kind="nysiis").encode_single(x_in) == x_out
