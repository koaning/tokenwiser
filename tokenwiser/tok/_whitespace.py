from tokenwiser.tok._tok import Tok


class WhiteSpaceTokenizer(Tok):
    def __init__(self):
        pass

    def encode_single(self, x):
        return [r for r in x.split(" ") if r != '']
