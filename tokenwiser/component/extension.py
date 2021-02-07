def sklearn_method(estimator):
    """
    A helper to turn a scikit-learn estimator into a spaCy extension.

    ```python
    import spacy
    from spacy.tokens import Doc

    from sklearn.pipeline import make_pipeline
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression

    from tokenwiser.component.extension import sklearn_method

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

    doc._.sillysent_method() # {0: 0.4446964938410244, 1: 0.5553035061589756}
    doc._.sillysent_prop # {0: 0.4446964938410244, 1: 0.5553035061589756}
    ```
    """

    def method(doc):
        proba = estimator.predict_proba([doc.text])[0]
        return {c: p for c, p in zip(estimator.classes_, proba)}

    return method
