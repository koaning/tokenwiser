import spacy 
from spacy.language import Language


def text_to_sentences(stream, nlp: Language, keep_orig_text=False):
    stream = ((ex['text'], ex) for ex in stream)
    for doc, ex in nlp.pipe(stream, as_tuples=True):
        if keep_orig_text:
            ex['orig_text'] = ex['text']
        for i, sent in enumerate(doc.sents):
            if keep_orig_text:
                yield {**ex, 'text': sent.text, "sent_idx": i}
            else:
                yield {**ex, "text": sent.text}

stream = ({"a": 1, "text": f"hello this is sentence number {i}. i am a dog?"} for i in range(100))
g = text_to_sentences(stream, spacy.load("en_core_web_md"))

next(g)