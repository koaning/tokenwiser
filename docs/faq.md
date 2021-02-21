## Why can't I use normal `Pipeline` objects with the spaCy API? 

Scikit-Learn assumes that data is trained via `.fit(X, y).predict(X)`. This is great
when you've got a dataset fully in memory but it's not so great when your dataset is 
too big to fit in one go. This is a main reason why spaCy has an `.update()`
API for their trainable pipeline components. It's similar to `.partial_fit(X)` in 
scikit-learn. You wouldn't train on a single batch of data. Instead you would iteratively
train on subsets of the dataset. 

A big downside of the `Pipeline` API is that it cannot use `.partial_fit(X)`. 
Even if all the components on the inside are compatible, it forces you to use `.fit(X)`. 
That is why this library offers a `PartialPipeline`. It only allows for components that have `.partial_fit` 
implemented and it's these pipelines that can also comply with spaCy's `.update()`
API.

Note that all scikit-learn components offered by this library are compatible with
the `PartialPipeline`. This includes everything from the `tokeniser.textprep` submodule. 

## Can I train spaCy with scikit-learn from Jupyter? 

It's not our favorite way of doing things, but nobody is stopping you. 


```python
import spacy 
from spacy import registry
from spacy.training import Example
from spacy.language import Language

from tokenwiser.pipeline import PartialPipeline
from tokenwiser.model.sklearnmod import SklearnCat
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

@Language.factory("sklearn-cat")
def make_sklearn_cat(nlp, name, sklearn_model, label, classes):
    return SklearnCat(nlp, name, sklearn_model, label, classes)

@registry.architectures("sklearn_model_basic_sgd.v1")
def make_sklearn_cat_basic_sgd():
    return PartialPipeline([("hash", HashingVectorizer()), ("lr", SGDClassifier(loss="log"))])


nlp = spacy.load("en_core_web_sm")
config = {
    "sklearn_model": "@sklearn_model_basic_sgd.v1", 
    "label": "pos", 
    "classes": ["pos", "neg"]
}
nlp.add_pipe("sklearn-cat", config=config)

texts = [
    "you are a nice person", 
    "this is a great movie", 
    "i do not like cofee", 
    "i hate tea"
]
labels = ["pos", "pos", "neg", "neg"]

# This is the training loop just for out categorizer model.
with nlp.select_pipes(enable="sklearn-cat"):
    optimizer = nlp.resume_training()
    for loop in range(10):
        for t, lab in zip(texts, labels):
            doc = nlp.make_doc(t)
            example = Example.from_dict(doc, {"cats": {"pos": lab}})
            nlp.update([example], sgd=optimizer)
            
nlp("you are a nice person").cats # {'pos': 0.9979167909733176}
nlp("coffee i do not like").cats # {'neg': 0.990049724779963}
```