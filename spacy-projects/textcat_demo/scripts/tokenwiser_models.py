from spacy import registry
from spacy.language import Language
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer

from tokenwiser.model import SklearnCat
from tokenwiser.pipeline import PartialPipeline


pipe = PartialPipeline([
    ("hash", HashingVectorizer()), 
    ("lr", SGDClassifier(loss="log"))
])

@Language.factory("sklearn-model")
def make_sklearn_cat(nlp, name, model, classes):
    return SklearnCat(nlp, name, sklearn_model=model, classes=classes)


@registry.architectures("sklearn_model_basic_sgd.v1")
def make_sklearn_cat_basic_sgd():
    return PartialPipeline(
        [("hash", HashingVectorizer()), ("lr", SGDClassifier(loss="log"))]
    )
