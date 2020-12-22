import numpy as np
from rich.progress import Progress
from sklearn.base import BaseEstimator
from tokenwiser.emb._emb import Emb
from tokenwiser.common import flatten

from nltk.corpus import wordnet


class Wordlap(BaseEstimator, Emb):
    def __init__(self, wordnet_depth=5):
        self.corpus = {}
        self.wordnet_depth = wordnet_depth

    def _add_words(self, word, depth=5):
        word = word.lower()
        if word in self.corpus.keys():
            return
        if depth == 0:
            return
        if "_" in word:
            for s in word.split("_"):
                self._add_words(s)
            return
        syn, ant = [], []
        for synset in wordnet.synsets(word):
            for lemma in synset.lemmas():
                syn.append(lemma.name())
                if lemma.antonyms():
                    ant.append(lemma.antonyms()[0].name())
        self.corpus[word] = {"ant": list(set(ant)), "syn": list(set(syn))}
        for w in set(ant + syn):
            self._add_words(w, depth=depth - 1)

    def _add_vocab(self, X):
        words = set(flatten(X))
        with Progress() as progress:
            task = progress.add_task("[green]Creating Wordnet...", total=len(words))
            for w in words:
                if "_" in w:
                    for _ in w.split("_"):
                        self._add_words(_)
                else:
                    self._add_words(w.lower(), depth=self.wordnet_depth)
                progress.update(task, advance=1)
        flat_toks = flatten(
            [v["ant"] + v["syn"] for v in self.corpus.values()] + [self.corpus.keys()]
        )
        self.token_lookup_ = {
            k: i
            for i, k in enumerate(
                set(flatten([_.lower().split("_") for _ in flat_toks]))
            )
        }

    def fit(self, X, y=None):
        self._add_vocab(X)
        return self

    @classmethod
    def from_file(cls):
        pass

    def transform(self, X, y=None):
        return np.array([self.encode_single(x) for x in X])

    def encode_single(self, x):
        pass

    def save(self, folder):
        pass

    @classmethod
    def from_folder(cls, folder):
        pass
