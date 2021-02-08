import pytest
import spacy
from spacy.training import Example
from tokenwiser.model.sklearnmod import SklearnCat


@pytest.mark.parametrize("model", ["@sklearn_model_basic_sgd.v1", "@sklearn_model_basic_naive_bayes.v1", "@sklearn_model_basic_pa.v1"])
def test_model_config_inline(model):
    nlp = spacy.load("en_core_web_sm")
    conf = {"sklearn_model": model, "label": "pos", "classes": ["pos", "neg"]}
    nlp.add_pipe("sklearn-cat", config=conf)

    texts = ["you are a nice person", "this is a great movie", "i do not like coffee"]
    labels = ["pos", "pos", "neg"]

    with nlp.select_pipes(enable="sklearn-cat"):
        optimizer = nlp.resume_training()
        for itn in range(100):
            for t, lab in zip(texts, labels):
                doc = nlp.make_doc(t)
                example = Example.from_dict(doc, {"cats": {"pos": lab}})
                nlp.update([example], sgd=optimizer)

    assert len(nlp("you are a nice person").cats.keys()) > 0
    assert len(nlp("coffee i do not like").cats.keys()) > 0
