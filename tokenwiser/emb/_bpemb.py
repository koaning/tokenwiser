from tokenwiser.emb._emb import Emb, Embedding
from bpemb import BPEmb


class BytePair(Emb):
    def __init__(self, lang="en", dim=300, vs=200_000):
        self.model = BPEmb(lang=lang, dim=dim, vs=vs)
        self.dim = dim

    @classmethod
    def from_file(cls):
        pass

    def encode_single(self, tokens):
        return [Embedding(name=t, vec=self.model.embed(t)) for t in tokens]
