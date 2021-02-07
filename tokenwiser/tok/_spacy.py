from tokenwiser.tok._tok import Tok

from sklearn.base import BaseEstimator


class SpacyTokenizer(Tok, BaseEstimator):
    """
    A tokenizer that uses spaCy under the hood for the tokenization.

    Arguments:
        model: reference to the spaCy model
        lemma: weather or not to also apply lemmatization
        stop: weather or not to remove stopwords

    Usage:

    ```python
    import spacy
    from tokenwiser.tok import SpacyTokenizer

    # This can also be a Non-English model.
    nlp = spacy.load("en_core_web_sm")
    tok = SpacyTokenizer(model=nlp)

    single = tok("hello world")
    assert single == ["hello", "world"]
    ```
    """

    def __init__(self, model, lemma=False, stop=False):
        self.model = model
        self.lemma = lemma
        self.stop = stop

    def __call__(self, text):
        if self.stop:
            return [
                t.lemma_ if self.lemma else t.text
                for t in self.model(text)
                if not t.is_stop
            ]
        return [t.lemma_ if self.lemma else t.text for t in self.model(text)]
