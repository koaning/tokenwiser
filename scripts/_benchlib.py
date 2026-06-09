"""Shared benchmark helpers for the two benchmark scripts.

This is a plain Python library — no command line. The two entry points
(``benchmark_quality.py`` and ``benchmark_scaling.py``) import these functions
directly.

Each training run happens in its own ``spawn``ed child process via
:func:`run_isolated`. That is purely for an honest **memory** measurement:
``resource.getrusage(...).ru_maxrss`` is a monotonic high-water mark for the
whole process and never decreases, so running multiple models — or a sweep of
growing dataset sizes — in one process would let earlier peaks pollute every
later reading. A fresh child per run keeps each peak-RSS number attributable to
that run alone. Real Python objects (tuples included) cross the process
boundary via pickle, so the tuple-vs-list search semantics are preserved.

Timing is always measured **from data-on-disk to a trained model**. Tokenwiser's
``train()`` reads the JSONL itself, so its ``total_seconds`` already includes
I/O; for scikit-learn we time the JSONL load and the fit separately and report
their sum. Everything runs single-threaded (``n_jobs=1``) — tokenwiser is
single-threaded, and child-process workers would otherwise hide their memory
from ``ru_maxrss``.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import resource
import sys
import time

# --- Fixed config for the scaling sweep: one CV pass, no real search. ------- #
SCALING_TW = dict(
    tokenizer="unicode",
    lowercase=True,
    ngram_range=(1, 1),
    n_features=20_000,
    n_hashes=2,
    feature_value="binary",
    epochs=3,
    learning_rate=0.1,
    l1=0.0,
    l2=1e-6,
)
SCALING_SK = {
    "tfidf__lowercase": [True],
    "tfidf__ngram_range": [(1, 1)],
    "tfidf__max_features": [20_000],
    "tfidf__binary": [True],
    "logreg__C": [1.0],
}

# --- Head-to-head search spaces (mirrored between the two libraries). ------- #
QUALITY_TW = dict(
    tokenizer=["unicode", "whitespace"],
    lowercase=True,
    ngram_range=[(1, 1), (1, 2)],
    n_features=(5_000, 100_000),
    n_hashes=(1, 4),
    feature_value=["binary", "count"],
    epochs=(3, 10),
    learning_rate=(1e-3, 3e-1),
    l1=0.0,
    l2=(1e-8, 1e-3),
    batch_size=256,
)
QUALITY_SK = {
    "tfidf__lowercase": [True],
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__max_features": [5_000, 10_000, 25_000, 50_000, 100_000],
    "tfidf__binary": [True, False],
    "logreg__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
}


def peak_rss_mb() -> float:
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports kibibytes.
    if sys.platform == "darwin":
        return maxrss / (1024 * 1024)
    return maxrss / 1024


def count_rows(path: str) -> int:
    n = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def load_jsonl(path: str):
    """Read a JSONL file into ``(texts, binary malicious labels)``."""
    import json

    texts, labels = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            texts.append(row["text"])
            labels.append(1 if "malicious" in (row.get("labels") or []) else 0)
    return texts, labels


def _metrics(y_true, y_pred, y_prob):
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def _base_result(model: str, train_path: str) -> dict:
    return {
        "model": model,
        "n_train_rows": count_rows(train_path),
        "load_seconds": None,
        "fit_seconds": None,
        "total_seconds": None,
        "best_cv_score": None,
        "best_config": None,
        "metrics": None,
        "predict_seconds": None,
        "n_test_rows": None,
        "model_bytes": None,
    }


def train_tokenwiser(
    train_path, *, trials, cv, seed, params, test_path=None, save_model=None
) -> dict:
    """Train (and optionally evaluate) tokenwiser. Runs inside a child process."""
    import tokenwiser

    result = _base_result("tokenwiser", train_path)
    t0 = time.perf_counter()
    model, report = tokenwiser.train(
        train_path, trials=trials, cv=cv, seed=seed, **params
    )
    result["total_seconds"] = time.perf_counter() - t0  # includes disk read
    result["best_cv_score"] = report["best_macro_f1"]
    result["best_config"] = report["best_config"]

    if test_path:
        texts, y = load_jsonl(test_path)
        p0 = time.perf_counter()
        probs = [row["malicious"] for row in model.predict_proba(texts)]
        result["predict_seconds"] = time.perf_counter() - p0
        result["metrics"] = _metrics(y, [int(p >= 0.5) for p in probs], probs)
        result["n_test_rows"] = len(y)

    if save_model:
        model.save(save_model)
        result["model_bytes"] = os.path.getsize(save_model)
    return result


def train_sklearn(
    train_path, *, trials, cv, seed, space, test_path=None, save_model=None
) -> dict:
    """Train (and optionally evaluate) the scikit-learn pipeline (child process)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
    from sklearn.pipeline import Pipeline

    result = _base_result("sklearn", train_path)

    l0 = time.perf_counter()
    texts, y = load_jsonl(train_path)
    result["load_seconds"] = time.perf_counter() - l0

    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            (
                "logreg",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1_000,
                    solver="liblinear",
                    random_state=seed,
                ),
            ),
        ]
    )
    search = RandomizedSearchCV(
        pipe,
        space,
        n_iter=trials,
        scoring="f1",
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed),
        random_state=seed,
        n_jobs=1,  # keep memory in-process and the comparison single-threaded
        refit=True,
    )
    f0 = time.perf_counter()
    search.fit(texts, y)
    result["fit_seconds"] = time.perf_counter() - f0
    result["total_seconds"] = result["load_seconds"] + result["fit_seconds"]
    result["best_cv_score"] = float(search.best_score_)
    result["best_config"] = {k: str(v) for k, v in search.best_params_.items()}

    if test_path:
        test_texts, test_y = load_jsonl(test_path)
        p0 = time.perf_counter()
        probs = search.predict_proba(test_texts)[:, 1].tolist()
        result["predict_seconds"] = time.perf_counter() - p0
        result["metrics"] = _metrics(test_y, [int(p >= 0.5) for p in probs], probs)
        result["n_test_rows"] = len(test_y)

    if save_model:
        import pickle

        with open(save_model, "wb") as fh:
            pickle.dump(search.best_estimator_, fh)
        result["model_bytes"] = os.path.getsize(save_model)
    return result


def _child(target, kwargs, queue):
    result = target(**kwargs)
    result["peak_rss_mb"] = peak_rss_mb()  # measured in the isolated child
    queue.put(result)


def run_isolated(target, **kwargs) -> dict:
    """Run ``target(**kwargs)`` in a fresh ``spawn`` child and return its result.

    The child measures its own peak RSS just before returning, so the memory
    figure belongs to this run alone.
    """
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_child, args=(target, kwargs, queue))
    proc.start()
    result = queue.get()
    proc.join()
    if proc.exitcode != 0:
        raise RuntimeError(f"isolated run failed (exit code {proc.exitcode})")
    return result
