from tokenwiser.tok._tok import Tok

from sklearn.base import BaseEstimator


class SpacyToken(Tok, BaseEstimator):
    def __init__(self, model, lemma=False, stop=False):
        self.model = model
        self.lemma = lemma
        self.stop = stop

    def encode_single(self, text):
        if self.stop:
            return [t.lemma_ if self.lemma else t.text for t in self.model(text) if not t.is_stop]
        return [t.lemma_ if self.lemma else t.text for t in self.model(text)]
