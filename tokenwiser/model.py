from typing import Any
import srsly
import numpy as np
from sklearn.linear_model import LogisticRegression
from wasabi import Printer
from icepickle.linear_model import save_coefficients, load_coefficients
from pathlib import Path
from sklearn.base import clone
from sklearn.metrics import classification_report
from spacy import Language 
from spacy.tokens import Span

msg = Printer()


class Categorizer:
    def __init__(self, encoder, tasks, mod=LogisticRegression(class_weight="balanced"), verbose=False):
        self._encoder = encoder
        self._tasks = tasks
        self._models = {k: clone(mod) for k in self._tasks}
        self._verbose = verbose

    def learn(self, examples):
        X = self._encoder.transform([ex["text"] for ex in examples])
        for task, model in self._models.items():
            idx = [i for i, ex in enumerate(examples) if task in ex]
            xs = X[idx]
            ys = np.array(
                [ex[task] for i, ex in enumerate(examples) if task in ex], dtype=int
            )
            model.fit(xs, ys)
            if self._verbose:
                msg.good(f"Trained the {task} task, using {len(ys)} examples.")

    def __call__(self, text):
        result = {}
        X = self._encoder.transform([text])
        for task in self._tasks:
            proba = self._models[task].predict_proba(X)[0, 1]
            result[task] = float(proba)
        return result

    def performance(self, examples):
        X = self._encoder.transform([ex["text"] for ex in examples])
        out = {}
        for task, model in self._models.items():
            idx = [i for i, ex in enumerate(examples) if task in ex]
            xs = X[idx]
            ys = np.array(
                [ex[task] for i, ex in enumerate(examples) if task in ex], dtype=int
            )
            out[task] = classification_report(ys, model.predict(xs), output_dict=True)
        return out

    def to_disk(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        for name, clf in self._models.items():
            save_coefficients(clf, Path(path) / f"{name}.h5")

    @classmethod
    def from_disk(cls, path, encoder):
        models = {}
        for f in Path(path).glob("*.h5"):
            clf_reloaded = LogisticRegression()
            load_coefficients(clf_reloaded, f)
            models[f.stem] = clf_reloaded

        model = Categorizer(encoder=encoder, tasks=models.keys())
        model._models = models
        return model
    

class SentenceModel:
    def __init__(self, nlp, categorzier, threshold):
        self._nlp = nlp
        self._categorizer: Categorizer = categorzier
        self._threshold = threshold
    
    def __call__(self, text):
        doc = self._nlp(text)
        for sent in doc.sents:
            preds = self.categorizer(sent.text)
            for label, proba in preds.items():
                if proba > self.threshold:
                    doc.spans["sc"].append(Span(doc, sent.start, sent.end, label))
                    doc.cats[label] = max(doc.cats.get(label, 0,0), proba)
        return doc
    
    def to_html(self, text, category, color="#FEF08A"):
        doc = self._nlp(text)
        out = ""
        for sent in doc.sents:
            proba = self._categorizer(text)[category]
            if proba > self._threshold:
                out += f"<span style='background-color: {color};'>{sent.text}</span>"
            else:
                out += f"<span>{sent.text}</span>"
        return out
