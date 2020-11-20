from tokenwiser.prep import HyphenPrep


def test_basic():
    assert HyphenPrep().encode_single("haleluja") == 'hale lu ja'
