from sklearn.base import BaseEstimator

from ._prep import TextPrep


class SpacyMorphTextPrep(TextPrep, BaseEstimator):
    """
    Adds morphologic information to tokens in text.

    Usage:

    ```python
    import spacy
    from tokenwiser.textprep import SpacyMorphTextPrep

    nlp = spacy.load("en_core_web_sm")
    example1 = SpacyMorphTextPrep(nlp).encode_single("quick! duck!")
    example2 = SpacyMorphTextPrep(nlp).encode_single("hey look a duck")

    assert example1 == "quick|Degree=Pos !|PunctType=Peri duck|Number=Sing !|PunctType=Peri"
    assert example2 == "hey| look|VerbForm=Inf a|Definite=Ind|PronType=Art duck|Number=Sing"
    ```
    """

    def __init__(self, model, lemma: bool =False):
        self.model = model
        self.lemma = lemma

    def encode_single(self, text):
        return " ".join(
            [
                f"{t.text if not self.lemma else t.lemma_}|{t.morph}"
                for t in self.model(text)
            ]
        )


class SpacyPosTextPrep(TextPrep, BaseEstimator):
    """
    Adds part of speech information per token using spaCy.

    Arguments:
        model: the spaCy model to use
        lemma: also lemmatize the text
        fine_grained: use fine grained parts of speech

    Usage:

    ```python
    import spacy
    from tokenwiser.textprep import SpacyPosTextPrep

    nlp = spacy.load("en_core_web_sm")
    example1 = SpacyPosTextPrep(nlp).encode_single("we need to duck")
    example2 = SpacyPosTextPrep(nlp).encode_single("hey look a duck")

    assert example1 == "we|PRON need|VERB to|PART duck|VERB"
    assert example2 == "hey|INTJ look|VERB a|DET duck|NOUN"
    ```
    """

    def __init__(self, model, lemma: bool = False, fine_grained: bool = False):
        self.model = model
        self.lemma = lemma
        self.fine_grained = fine_grained

    def encode_single(self, text):
        return " ".join(
            [
                f"{t.text if not self.lemma else t.lemma_}|{t.tag_ if self.fine_grained else t.pos_}"
                for t in self.model(text)
            ]
        )


class SpacyLemmaTextPrep(TextPrep, BaseEstimator):
    """
    Turns each token into a lemmatizer version using spaCy.

    Usage:

    ```python
    import spacy
    from tokenwiser.textprep import SpacyLemmaTextPrep

    nlp = spacy.load("en_core_web_sm")
    example1 = SpacyLemmaTextPrep(nlp).encode_single("we are running")
    example2 = SpacyLemmaTextPrep(nlp).encode_single("these are dogs")

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

    def transform(self, X, y=None):
        return [" ".join([t.lemma_ for t in d]) for d in self.model.pipe(X)]
