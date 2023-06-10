import srsly 
from skops.io import dump, load
from pathlib import Path
from typing import List
import itertools as it 

class SentenceModel:
    def __init__(self, encoder, classifier, categories: List[str], trained_heads=None):
        self._encoder = encoder
        self._classifier = classifier
        self._categories = categories
        self._trained_heads = {} if not trained_heads else trained_heads
    
    def learn(self, stream):
        for cat in self._categories:
            new_stream, stream = it.tee(stream)
            
            # Only get the relevant datapoints
            curr_stream = ((_['text'], _[cat]) for _ in new_stream if cat in _.keys())
            texts, labels = zip(*curr_stream)
            X = self._encoder.transform(list(texts))
            y = list(labels)

            # Train a model
            self._trained_heads[cat] = self._classifier.copy().fit(X, y)
    
    def to_disk(self, path: Path):
        assert path.is_dir()
        path.mkdir(parents=True, exist_ok=True)
        for name, clf in self._trained_heads.items():
            dump(clf, path / f"{name}.model")
    
    @classmethod
    def from_disk(cls, path: Path, encoder):
        trained_heads = {}
        categories = []
        for mod_path in path.glob("*.model"):
            categories.append(mod_path.stem)
            trained_heads[mod_path.stem] = load(mod_path, trusted=True)
        return SentenceModel(encoder=encoder, 
                             classifier=trained_heads[mod_path.stem].copy(), 
                             categories=categories,
                             trained_heads=trained_heads)
    
    def __call__(self, stream, pred_key=None):
        for ex in stream:
            predictions = {}
            for cat in self._categories:
                predictions[cat] = self._trained_heads.predict([ex['text']])
            if pred_key:
                yield {**ex, pred_key: predictions}
            else:
                yield {**ex, **predictions}
            