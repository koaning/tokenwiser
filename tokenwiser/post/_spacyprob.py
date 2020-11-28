import spacy
import numpy as np

from ._post import Post
from tokenwiser.emb._emb import Embedding


class SpacyProb(Post):
    def __init__(self, spacy_mod, alpha=1.0):
        if isinstance(spacy_mod, str):
            self.model = spacy.load(spacy_mod)
        else:
            self.model = spacy_mod
        if self.model.vocab.lookups_extra.has_table("lexeme_prob"):
            self.model.vocab.lookups_extra.remove_table("lexeme_prob")
        self.alpha = alpha

    def encode_single(self, embs):
        weights = [
            (1 - self.alpha) + self.alpha / np.exp(self.model.vocab[e.name].prob)
            for e in embs
        ]
        weights = self.alpha * np.array(weights) / sum(weights) + (
            1 - self.alpha
        ) * np.ones(len(weights))
        return [Embedding(name=e.name, vec=e.vec * w) for w, e in zip(weights, embs)]
