from tokenwiser.emb._emb import Emb, Embedding
import fasttext


class FastText(Emb):
    def __init__(self, model):
        self.model = fasttext.load_model(model)

    @classmethod
    def from_file(cls):
        pass

    def encode_single(self, tokens):
        return [Embedding(name=t, vec=self.model.get_word_vector(t)) for t in tokens]
