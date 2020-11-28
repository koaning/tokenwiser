import spacy
import numpy as np

from ._post import Post
from tokenwiser.emb._emb import Embedding


class SpacyProb(Post):
    def __init__(self, spacy_mod):
        if isinstance(spacy_mod, str):
            self.model = spacy.load(spacy_mod)
        else:
            self.model = spacy_mod

    def encode_single(self, embs):
        weights = [1 / np.exp(self.model.vocab[e.name]) for e in embs]
        weights = np.array(weights) / sum(weights)
        return [Embedding(name=e.name, vec=e.vec * w) for w, e in zip(weights, embs)]
