import json

import pytest

# A small, linearly separable multilabel dataset:
#   - "pos" vs "neg" sentiment words
#   - an orthogonal "emotion" label
_ROWS = [
    {"text": "good great excellent wonderful", "labels": ["pos"]},
    {"text": "bad terrible awful horrible", "labels": ["neg"]},
    {"text": "great wonderful amazing fantastic", "labels": ["pos"]},
    {"text": "awful horrible disgusting bad", "labels": ["neg"]},
    {"text": "love happy joy great", "labels": ["pos", "emotion"]},
    {"text": "hate angry sad terrible", "labels": ["neg", "emotion"]},
    {"text": "excellent superb great love", "labels": ["pos", "emotion"]},
    {"text": "worst awful hate angry", "labels": ["neg", "emotion"]},
    {"text": "neutral plain ordinary boring", "labels": []},
    {"text": "good nice pleasant lovely", "labels": ["pos"]},
    {"text": "bad nasty unpleasant grim", "labels": ["neg"]},
    {"text": "fantastic brilliant great joy", "labels": ["pos", "emotion"]},
]


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "train.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for row in _ROWS:
            fh.write(json.dumps(row) + "\n")
    return str(path)


@pytest.fixture
def fast_kwargs():
    """Small, fully-fixed config so tests are fast and deterministic."""
    return dict(
        trials=4,
        cv=3,
        seed=7,
        n_features=2048,
        n_hashes=2,
        epochs=20,
        learning_rate=0.5,
        l2=1e-6,
    )
