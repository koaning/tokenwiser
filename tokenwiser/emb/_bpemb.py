from tokenwiser.emb._emb import Emb, Embedding
from bpemb import BPEmb


class BytePair(Emb):
    def __init__(self, lang="en", dim=300, vs=200_000):
        self.model = BPEmb(lang=lang, dim=dim, vs=vs)
        self.dim = dim

    @classmethod
    def from_file(cls):
        pass

    def transform(self, tokens, y=None):
        return [self.encode_single(t) for t in tokens]

    def encode_single(self, token):
        return Embedding(name=token, vec=self.model.embed(token))
