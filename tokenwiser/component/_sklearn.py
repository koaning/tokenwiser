from spacy.language import Language


def attach_sklearn_categoriser(nlp, pipe_name, estimator):
    """
    This function will attach a scikit-learn compatible estimator to
    the pipeline which will feed predictions to the `.cats` property.
    """

    @Language.component(pipe_name)
    def my_component(doc):
        pred = estimator.predict([doc.text])[0]
        proba = estimator.predict_proba([doc.text]).max()
        doc.cats[pred] = proba
        return doc

    nlp.add_pipe(pipe_name)


def attach_sklearn_attribute(nlp, pipe_name, attribute_name, estimator):
    """
    This function will attach an attribute to the prediction.
    """

    @Language.component(pipe_name)
    def my_component(doc):
        pred = estimator.predict([doc.text])[0]
        setattr(doc, attribute_name, pred)
        return doc

    nlp.add_pipe(pipe_name)
