from sklearn.base import BaseEstimator

from tokenwiser.prep._prep import Prep


class SpacyMorphPrep(Prep, BaseEstimator):
    """Adds morphologic information to tokens in text.

    Usage:

    ```python
    from tokenwiser.prep import SpacyMorphPrep

    nlp = spacy.load("en_core_web_sm")
    example1 = SpacyMorphPrep(nlp).encode_single("quick! duck!")
    example2 = SpacyMorphPrep(nlp).encode_single("hey look a duck")

    assert example1 == "quick|Degree=Pos !|PunctType=Peri duck|Number=Sing !|PunctType=Peri"
    assert example2 == "hey| look|VerbForm=Inf a|Definite=Ind|PronType=Art duck|Number=Sing"
    ```
    """
    def __init__(self, model):
        self.model = model

    def encode_single(self, text):
        return " ".join([f"{t.text}|{t.morph}" for t in self.model(text)])


class SpacyPosPrep(Prep, BaseEstimator):
    """
    Adds part of speech information per token using spaCy.

    Usage:

    ```python
    from tokenwiser.prep import SpacyMorphPrep

    nlp = spacy.load("en_core_web_sm")
    example1 = SpacyPosPrep(nlp).encode_single("we need to duck")
    example2 = SpacyPosPrep(nlp).encode_single("hey look a duck")

    assert example1 == "we|PRON need|VERB to|PART duck|VERB"
    assert example2 == "hey|INTJ look|VERB a|DET duck|NOUN"
    ```
    """
    def __init__(self, model):
        self.model = model

    def encode_single(self, text):
        return " ".join([f"{t.text}|{str(t.pos_)}" for t in self.model(text)])


class SpacyLemmaPrep(Prep, BaseEstimator):
    """
    Turns each token into a lemmatizer version using spaCy.

    Usage:

    ```python
    from tokenwiser.prep import SpacyMorphPrep

    nlp = spacy.load("en_core_web_sm")
    example1 = SpacyLemmaPrep(nlp).encode_single("we are running")
    example2 = SpacyLemmaPrep(nlp).encode_single("these are dogs")

    assert example1 == 'we be run'
    assert example2 == 'these be dog'
    ```
    """
    def __init__(self, model, stop=False):
        self.stop = stop
        self.model = model

    def encode_single(self, text):
        if self.stop:
            return " ".join([t.lemma_ for t in self.model(text) if not t.is_stop])
        return " ".join([t.lemma_ for t in self.model(text)])
