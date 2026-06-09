"""tokenwiser: a Rust-backed sparse multilabel text classifier.

The public API is intentionally tiny::

    model, report = tokenwiser.train("train.jsonl")
    labels        = model.predict(["some text"])
    probabilities = model.predict_proba(["some text"])
    model.save("model.tw")

    model  = tokenwiser.load("model.tw")
    report.to_dict()
    report.save("report.json")

Search-space semantics for the ``train`` keyword arguments:

* **scalar** -> fixed value
* **list**   -> discrete choices
* **tuple**  -> a randomized range (semantics depend on the parameter)

``ngram_range=(1, 2)`` is a *fixed* structured tuple; ``[(1, 1), (1, 2)]`` is a
choice. ``epochs=(2, 10)`` is an inclusive integer range; ``learning_rate``,
``l1`` and ``l2`` ranges are log-uniform. Class weighting is always balanced,
randomized search optimizes cross-validated macro F1, and prediction uses a
fixed 0.5 threshold.
"""

import copy
import json

from . import _core

__version__ = "1.0.0"

__all__ = ["train", "load", "Model", "TrainingReport"]

_TOKENIZERS = {"unicode", "whitespace"}
_FEATURE_VALUES = {"binary", "count"}


# --------------------------------------------------------------------------- #
# Type-aware parameter parsing
# --------------------------------------------------------------------------- #
def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _is_number(x):
    return _is_int(x) or isinstance(x, float)


def _str_spec(value, name, allowed):
    def check(v):
        if not isinstance(v, str):
            raise ValueError(f"{name}: expected string(s), got {v!r}")
        if v not in allowed:
            raise ValueError(f"{name}: {v!r} is not one of {sorted(allowed)}")

    if isinstance(value, str):
        check(value)
        return {"type": "fixed", "value": value}
    if isinstance(value, list):
        if not value:
            raise ValueError(f"{name}: choice list must not be empty")
        for v in value:
            check(v)
        return {"type": "choice", "values": list(value)}
    raise ValueError(f"{name}: expected a string or a list of strings, got {value!r}")


def _bool_spec(value, name):
    if isinstance(value, bool):
        return {"type": "fixed", "value": value}
    if isinstance(value, list):
        if not value:
            raise ValueError(f"{name}: choice list must not be empty")
        for v in value:
            if not isinstance(v, bool):
                raise ValueError(f"{name}: expected booleans, got {v!r}")
        return {"type": "choice", "values": list(value)}
    raise ValueError(f"{name}: expected a bool or a list of bools, got {value!r}")


def _int_spec(value, name, min_value=1):
    def check(v):
        if not _is_int(v):
            raise ValueError(f"{name}: expected integer(s), got {v!r}")
        if v < min_value:
            raise ValueError(f"{name}: must be >= {min_value}, got {v}")

    if _is_int(value):
        check(value)
        return {"type": "fixed", "value": value}
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(
                f"{name}: an integer range must be a (lo, hi) tuple, got {value!r}"
            )
        lo, hi = value
        check(lo)
        check(hi)
        if lo > hi:
            raise ValueError(f"{name}: range lo {lo} > hi {hi}")
        return {"type": "int_range", "lo": lo, "hi": hi}
    if isinstance(value, list):
        if not value:
            raise ValueError(f"{name}: choice list must not be empty")
        for v in value:
            check(v)
        return {"type": "choice", "values": list(value)}
    raise ValueError(
        f"{name}: expected an int, a (lo, hi) tuple, or a list of ints, got {value!r}"
    )


def _float_spec(value, name):
    def check_fixed(v):
        if not _is_number(v):
            raise ValueError(f"{name}: expected number(s), got {v!r}")
        if v < 0:
            raise ValueError(f"{name}: must be >= 0, got {v}")

    if _is_number(value):
        check_fixed(value)
        return {"type": "fixed", "value": float(value)}
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(
                f"{name}: a log range must be a (lo, hi) tuple, got {value!r}"
            )
        lo, hi = value
        if not (_is_number(lo) and _is_number(hi)):
            raise ValueError(f"{name}: range bounds must be numbers, got {value!r}")
        if lo <= 0 or hi <= 0:
            raise ValueError(
                f"{name}: log-uniform ranges cannot include zero or negatives; "
                f"use a fixed value or a list choice instead"
            )
        if lo > hi:
            raise ValueError(f"{name}: range lo {lo} > hi {hi}")
        return {"type": "log_range", "lo": float(lo), "hi": float(hi)}
    if isinstance(value, list):
        if not value:
            raise ValueError(f"{name}: choice list must not be empty")
        for v in value:
            check_fixed(v)
        return {"type": "choice", "values": [float(v) for v in value]}
    raise ValueError(
        f"{name}: expected a number, a (lo, hi) tuple, or a list of numbers, "
        f"got {value!r}"
    )


def _ngram_spec(value, name):
    def check(t):
        if not (isinstance(t, tuple) and len(t) == 2):
            raise ValueError(f"{name}: each n-gram range must be a (min, max) tuple")
        lo, hi = t
        if not (_is_int(lo) and _is_int(hi)):
            raise ValueError(f"{name}: n-gram bounds must be integers, got {t!r}")
        if lo < 1 or hi < lo:
            raise ValueError(f"{name}: require 1 <= min <= max, got {t!r}")

    if isinstance(value, tuple):
        check(value)
        return {"type": "fixed", "value": [value[0], value[1]]}
    if isinstance(value, list):
        if not value:
            raise ValueError(f"{name}: choice list must not be empty")
        for t in value:
            check(t)
        return {"type": "choice", "values": [[t[0], t[1]] for t in value]}
    raise ValueError(
        f"{name}: expected a (min, max) tuple or a list of such tuples, got {value!r}"
    )


def _plain_int(value, name, min_value):
    if not _is_int(value):
        raise ValueError(f"{name}: expected a fixed integer, got {value!r}")
    if value < min_value:
        raise ValueError(f"{name}: must be >= {min_value}, got {value}")
    return value


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def train(
    path,
    *,
    text_field="text",
    labels_field="labels",
    trials=12,
    cv=3,
    seed=42,
    tokenizer="unicode",
    lowercase=True,
    ngram_range=(1, 1),
    n_features=(10_000, 100_000),
    n_hashes=(1, 3),
    feature_value="binary",
    epochs=(3, 10),
    learning_rate=(1e-3, 3e-1),
    l1=0.0,
    l2=(1e-8, 1e-3),
    batch_size=256,
):
    """Train a multilabel classifier from a JSONL file.

    Returns a ``(Model, TrainingReport)`` tuple. See the module docstring for
    the scalar/list/tuple search-space semantics.
    """
    spec = {
        "text_field": str(text_field),
        "labels_field": str(labels_field),
        "trials": _plain_int(trials, "trials", 1),
        "cv": _plain_int(cv, "cv", 2),
        "seed": _plain_int(seed, "seed", 0),
        "batch_size": _plain_int(batch_size, "batch_size", 1),
        "tokenizer": _str_spec(tokenizer, "tokenizer", _TOKENIZERS),
        "lowercase": _bool_spec(lowercase, "lowercase"),
        "ngram_range": _ngram_spec(ngram_range, "ngram_range"),
        "n_features": _int_spec(n_features, "n_features", min_value=1),
        "n_hashes": _int_spec(n_hashes, "n_hashes", min_value=1),
        "feature_value": _str_spec(feature_value, "feature_value", _FEATURE_VALUES),
        "epochs": _int_spec(epochs, "epochs", min_value=1),
        "learning_rate": _float_spec(learning_rate, "learning_rate"),
        "l1": _float_spec(l1, "l1"),
        "l2": _float_spec(l2, "l2"),
    }

    core_model, report_json = _core.train(json.dumps(spec), str(path))
    return Model(core_model), TrainingReport(json.loads(report_json))


def load(path):
    """Load a model previously written with :meth:`Model.save`."""
    return Model(_core.load(str(path)))


def _as_texts(texts):
    if isinstance(texts, str):
        return [texts]
    return [str(t) for t in texts]


class Model:
    """A trained classifier. Construct via :func:`train` or :func:`load`."""

    def __init__(self, core):
        self._core = core

    @property
    def labels(self):
        """The label vocabulary, in column order."""
        return self._core.labels()

    def predict(self, texts):
        """Return the predicted labels for each text (0.5 threshold)."""
        return self._core.predict(_as_texts(texts))

    def predict_proba(self, texts):
        """Return a ``{label: probability}`` dict for each text."""
        labels = self._core.labels()
        rows = self._core.predict_proba(_as_texts(texts))
        return [dict(zip(labels, row)) for row in rows]

    def save(self, path):
        """Write the model to ``path`` as a versioned ``.tw`` artifact."""
        self._core.save(str(path))

    def __repr__(self):
        return f"Model(labels={len(self._core.labels())})"


class TrainingReport:
    """Structured record of a training run."""

    def __init__(self, data):
        self._data = data

    def to_dict(self):
        """Return a deep copy of the report as a plain dict."""
        return copy.deepcopy(self._data)

    def save(self, path):
        """Write the report to ``path`` as JSON."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self._data, fh, indent=2)

    def __getitem__(self, key):
        return self._data[key]

    def __repr__(self):
        return (
            f"TrainingReport(n_rows={self._data['n_rows']}, "
            f"n_labels={self._data['n_labels']}, "
            f"best_trial={self._data['best_trial']}, "
            f"best_macro_f1={self._data['best_macro_f1']:.4f})"
        )
