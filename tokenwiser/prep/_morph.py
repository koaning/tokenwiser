from sklearn.base import BaseEstimator

from tokenwiser.prep._prep import Prep


class SpacyMorphPrep(Prep, BaseEstimator):
    def __init__(self, model):
        self.model = model

    def encode_single(self, text):
        return " ".join([str(t.morph) for t in self.model(text)])


class SpacyTextPrep(Prep, BaseEstimator):
    def __init__(self, model, lemma=False, stop=False):
        self.lemma = lemma
        self.stop = stop
        self.model = model

    def encode_single(self, text):
        if self.stop:
            return " ".join([t.lemma_ if self.lemma else t.text for t in self.model(text) if not t.is_stop])
        return " ".join([t.lemma_ if self.lemma else t.text for t in self.model(text)])
