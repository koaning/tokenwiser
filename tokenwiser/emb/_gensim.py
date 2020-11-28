import pathlib
import yaml

import numpy as np
from rich.progress import Progress
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec
from sklearn.base import BaseEstimator
from tokenwiser.emb._emb import Emb, Embedding


class RichProgress(CallbackAny2Vec):
    def __init__(self, progress, task):
        self.epoch = 0
        self.progress = progress
        self.task = task

    def on_epoch_begin(self, model):
        pass

    def on_epoch_end(self, model):
        self.progress.update(self.task, advance=1)


class Gensim(Emb, BaseEstimator):
    def __init__(
        self, dim=100, window=5, min_count=1, workers=4, epochs=10, learning_rate=0.01
    ):
        self.dim = dim
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.learning_rate = learning_rate

    def fit(self, X, y=None):
        self.model = Word2Vec(
            size=self.dim,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            alpha=self.learning_rate,
        )
        self.model.build_vocab(X)
        with Progress() as progress:
            task = progress.add_task("[green]Training Gensim...", total=self.epochs)
            self.model.train(
                X,
                total_examples=self.model.corpus_count,
                epochs=self.epochs,
                callbacks=[RichProgress(progress, task)],
            )
        return self

    def fit_partial(self, X):
        if not self.model:
            return self.fit(X)
        self.model.build_vocab(X, update=True)
        self.model.train(X, total_examples=self.model.corpus_count, epochs=self.epochs)
        return self

    @classmethod
    def from_file(cls):
        pass

    def transform(self, X, y=None):
        return [self.encode_single(x) for x in X]

    def encode_single(self, tokens):
        if len(tokens) == 0:
            return [Embedding(name="", vec=np.zeros(self.dim))]
        vecs = [
            self.model.wv[t] if (t in self.model.wv) else np.zeros(self.dim)
            for t in tokens
        ]
        return [Embedding(name=t, vec=v) for t, v in zip(tokens, vecs)]

    def save(self, folder):
        self.model.wv.save(f"{folder}/wordvectors.kv")

    @classmethod
    def from_folder(cls, folder):
        conf = yaml.load(
            (pathlib.Path(folder) / "config.yml").read_text(), Loader=yaml.FullLoader
        )
        params = [p for p in conf["pipeline"] if p["name"] == "Gensim"][0]
        res = Gensim(**{k: v for k, v in params.items() if k != "name"})
        res.model = Word2Vec(
            size=params["dim"],
            window=params["window"],
            min_count=params["min_count"],
            workers=params["workers"],
            alpha=params["learning_rate"],
        )
        res.model.wv = KeyedVectors.load(f"{folder}/wordvectors.kv")
        return res
