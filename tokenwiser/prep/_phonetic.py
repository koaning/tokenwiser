import jellyfish

from tokenwiser.prep._prep import Prep


class PhoneticPrep(Prep):
    def __init__(self, kind="soundex"):
        methods = {"soundex": jellyfish.soundex,
                   "metaphone": jellyfish.metaphone,
                   "nysiis": jellyfish.nysiis}
        self.method = methods[kind]

    def encode_single(self, x):
        return " ".join([self.method(d) for d in x.split(" ")])
