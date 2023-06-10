from typing import Generator 
from spacy.language import Language


def to_sentences(stream: Generator, nlp: Language, keep_orig_text:bool=False):
    """
    Turns a stream of "text" keys into a new stream where each example
    resembles a sentence from the original text. Uses spaCy to split
    the sentences. 

    Arguments:
        stream: the original stream
        nlp: the spaCy model
        keep_orig_text: flag to keep the original text around under the "orig_text" key
    """
    stream = ((ex['text'], ex) for ex in stream)
    for doc, ex in nlp.pipe(stream, as_tuples=True):
        if keep_orig_text:
            ex['orig_text'] = ex['text']
        for i, sent in enumerate(doc.sents):
            if keep_orig_text:
                yield {**ex, 'text': sent.text, "sent_idx": i}
            else:
                yield {**ex, "text": sent.text}


