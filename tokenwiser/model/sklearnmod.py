import random
from typing import Iterable

import spacy
from spacy.tokens import Doc
from spacy.training import Example
from spacy.language import Language
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from joblib import dump, load

from tokenwiser.pipeline import PartialPipeline

@Language.factory("sklearn-cat")
def make_sklearn_cat(nlp, name, sklearn_model, label, classes):
    return SklearnCat(nlp, name, sklearn_model, label, classes)


class SklearnCat:
    def __init__(self, nlp, name, sklearn_model, label, classes):
        self.nlp = nlp
        self.name = name
        self.label = label
        self.classes = classes
        self.sklearn_model = spacy.registry.architectures.get(sklearn_model.replace("@", ""))()

    def __call__(self, doc: Doc):
        scores = self.predict([doc])
        self.set_annotations([doc], scores)
        return doc

    def update(self, examples: Iterable[Example], *, drop: float=0.0, sgd=None, losses=None):
        texts = [ex.reference.text for ex in examples if self.label in ex.reference.cats.keys()]
        labels = [ex.reference.cats[self.label] for ex in examples if self.label in ex.reference.cats.keys()]
        self.sklearn_model.partial_fit(texts, labels, classes=self.classes)

    def predict(self, docs: Iterable[Doc]):
        return self.sklearn_model.predict_proba([d.text for d in docs]).max(axis=1)

    def set_annotations(self, docs: Iterable[Doc], scores):
        preds = self.sklearn_model.predict([d.text for d in docs])
        for doc, pred, proba in zip(docs, preds, scores):
            doc.cats[pred] = proba
        return docs

    def score(self):
        return random.random()

    def to_disk(self, path):
        pass

    def from_disk(self, path):
        pass


@spacy.registry.architectures("sklearn_model_basic.v1")
def make_sklearn_cat_basic():
    return PartialPipeline([("hash", HashingVectorizer()), ("lr", SGDClassifier(loss="log"))])