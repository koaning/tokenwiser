from spacy.tokens import Doc, Token

from tokenwiser.textprep import HyphenTextPrep


def attach_hyphen_extension():
    """
    This function will attach an extension `._.hyphen` to the `Token`s.

    ```python
    import spacy
    from tokenwiser.extension import attach_hyphen_extension

    nlp = spacy.load("en_core_web_sm")
    # Attach the Hyphen extensions.
    attach_hyphen_extension()

    # Now you can query hyphens on the tokens.
    doc = nlp("this is a dinosaurhead")
    tok = doc[-1]

    assert tok._.hyphen == ["di", "no", "saur", "head"]
    ```
    """
    Token.set_extension(
        "hyphen",
        getter=lambda t: HyphenTextPrep().encode_single(t.text).split(" "),
        force=True,
    )


def attach_sklearn_extension(attribute_name, estimator):
    """
    This function will attach an extension `._.attribute_name` to the `Token`s.

    ```python
    import spacy
    from spacy.tokens import Doc

    from sklearn.pipeline import make_pipeline
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression

    from tokenwiser.extension import attach_sklearn_extension

    X = [
        "i really like this post",
        "thanks for that comment",
        "i enjoy this friendly forum",
        "this is a bad post",
        "i dislike this article",
        "this is not well written"
    ]

    y = ["pos", "pos", "pos", "neg", "neg", "neg"]

    # First we train a (silly) model.
    mod = make_pipeline(CountVectorizer(), LogisticRegression()).fit(X, y)

    # Demo
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("thank you, really nice")
    attach_sklearn_extension("sillysent", mod)
    doc._.sillysent # {"neg: 0.4446964938410244, "pos": 0.5553035061589756}
    ```
    """
    Doc.set_extension(
        attribute_name,
        getter=lambda t: sklearn_method(estimator=estimator),
        force=True,
    )


def sklearn_method(estimator):
    """
    A helper to turn a scikit-learn estimator into a spaCy extension.

    Just in case you *really* wanted to do it manually.

    ```python
    import spacy
    from spacy.tokens import Doc

    from sklearn.pipeline import make_pipeline
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression

    from tokenwiser.extension import sklearn_method

    X = [
        "i really like this post",
        "thanks for that comment",
        "i enjoy this friendly forum",
        "this is a bad post",
        "i dislike this article",
        "this is not well written"
    ]

    y = ["pos", "pos", "pos", "neg", "neg", "neg"]

    # First we train a (silly) model.
    mod = make_pipeline(CountVectorizer(), LogisticRegression()).fit(X, y)

    # This is where we attach the scikit-learn model to spaCy as a method extension.
    Doc.set_extension("sillysent_method", method=sklearn_method(mod))
    # This is where we attach the scikit-learn model to spaCy as a property extension.
    Doc.set_extension("sillysent_prop", getter=sklearn_method(mod))

    # Demo
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("thank you, really nice")

    doc._.sillysent_method() # {"neg": 0.4446964938410244, "pos: 0.5553035061589756}
    doc._.sillysent_prop # {"neg: 0.4446964938410244, "pos": 0.5553035061589756}
    ```
    """

    def method(doc):
        proba = estimator.predict_proba([doc.text])[0]
        return {c: p for c, p in zip(estimator.classes_, proba)}

    return method
