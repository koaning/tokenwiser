import pyphen
from tokenwiser.prep._prep import Prep


class HyphenPrep(Prep):
    def __init__(self, lang="en_GB"):
        self.dic = pyphen.Pyphen(lang=lang)

    def encode_single(self, x):
        return " ".join(self.dic.inserted(x).split("-", -1))