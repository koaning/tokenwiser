import random
import pathlib
from typing import Iterable

import spacy
from spacy import registry
from spacy.tokens import Doc
from spacy.training import Example
from spacy.language import Language
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from joblib import dump, load

from tokenwiser.pipeline import PartialPipeline


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

    def to_disk(self, path, exclude=None):
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        dump(self.sklearn_model, str(pathlib.Path(path)/'filename.joblib'))

    def from_disk(self, path, exclude=None):
        self.sklearn_model = load(str(pathlib.Path(path)/'filename.joblib'))
        return self


@Language.factory("sklearn-cat")
def make_sklearn_cat(nlp, name, sklearn_model, label, classes):
    return SklearnCat(nlp, name, sklearn_model, label, classes)


@registry.architectures("sklearn_model_basic_sgd.v1")
def make_sklearn_cat_basic_sgd():
    return PartialPipeline([("hash", HashingVectorizer()), ("lr", SGDClassifier(loss="log"))])


@registry.architectures("sklearn_model_basic_naive_bayes.v1")
def make_sklearn_cat_basic_naive_bayes():
    return PartialPipeline([("hash", HashingVectorizer(binary=True)), ("nb", MultinomialNB())])


@registry.architectures("sklearn_model_basic_pa.v1")
def make_sklearn_cat_basic_naive_bayes():
    return PartialPipeline([("hash", HashingVectorizer(binary=True)), ("pa", PassiveAggressiveClassifier())])
