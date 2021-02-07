from typing import Any

import spacy
from spacy.vocab import Vocab
from spacy.language import Language
from spacy.pipeline import TrainablePipe
from thinc.types import ArgsKwargs
from thinc.model import Shim, Model
from vowpalwabbit.sklearn_vw import VWClassifier


class VowpalWabbitShim(Shim):
    """
    Interface between a Vowpal Wabbit model and a Thinc Model.

    This container is *not* a Thinc Model subclass itself.
    """

    def __init__(self):
        self.vw = VWClassifier()

    def __call__(self, inputs, is_train):
        if is_train:
            return self.begin_update(inputs)
        else:
            return self.predict(inputs), lambda a: ...

    def predict(self, inputs: ArgsKwargs) -> Any:
        """
        Make a prediction using VowpalWabbit
        """
        print(inputs)
        return self.vw.predict(inputs["X"])

    def begin_update(self, inputs: ArgsKwargs):
        print(inputs)
        return self.predict(inputs=inputs), lambda a: ...

    def finish_update(self, optimizer):
        pass


@spacy.registry.layers("VowpalWabbitWrapper.v1")
def VowpalWabbitWrapper():
    return Model("vowpalwabbit", forward_vw, shims=[VowpalWabbitShim()])


def forward_vw(model: Model, X: Any, is_train: bool):
    vw_clf = model.shims[0]
    return vw_clf.predict(X), lambda a: ...


class VowpalWabbitTrainablePipe(TrainablePipe):
    def __init__(self, vocab: Vocab, model: Model, name: str, **cfg):
        self.vocab = vocab
        self.model = model
        self.name = name
        self.cfg = dict(cfg)

    def predict(self, docs):
        return self.model.shims[0].predict([d.text for d in docs])

    def set_annotations(self, docs, scores):
        preds = self.predict(docs)
        for doc, pred in zip(docs, preds):
            doc.cats[pred] = pred


@Language.factory("my_trainable_component")
def make_component(nlp, name, model):
    return VowpalWabbitTrainablePipe(nlp.vocab, model, name=name)
