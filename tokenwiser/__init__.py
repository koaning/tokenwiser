from typing import List
import itertools as it 

class SentenceModel:
    def __init__(self, encoder, classifier, categories: List[str]):
        self._encoder = encoder
        self._classifier = classifier
        self._categories = categories
        self._trained_heads = {}
    
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
    
    def __call__(self, stream, pred_key=None):
        for ex in stream:
            predictions = {}
            for cat in self._categories:
                predictions[cat] = self._trained_heads.predict([ex['text']])
            if pred_key:
                yield {**ex, pred_key: predictions}
            else:
                yield {**ex, **predictions}
            