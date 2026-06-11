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
import threading
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


class PeakTreeRSS:
    """Sample peak RSS of this process + all children, on a background thread.

    Used so parallel runs (sklearn's n_jobs=-1 spawns worker processes) are
    measured the same way as single-process runs — `ru_maxrss` only sees the
    calling process and would hide the workers' memory.
    """

    def __init__(self, interval=0.03):
        import psutil

        self._proc = psutil.Process()
        self._interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread = None

    def _sample(self):
        import psutil

        while not self._stop.is_set():
            try:
                total = self._proc.memory_info().rss
                for c in self._proc.children(recursive=True):
                    try:
                        total += c.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                self.peak = max(self.peak, total)
            except Exception:
                pass
            self._stop.wait(self._interval)

    def __enter__(self):
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2)

    @property
    def peak_mb(self):
        return self.peak / 1e6


def _child(target, kwargs, queue):
    result = target(**kwargs)
    # Targets that measure their own (tree) peak keep it; others fall back to
    # the kernel's single-process high-water mark.
    result.setdefault("peak_rss_mb", peak_rss_mb())
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


# --------------------------------------------------------------------------- #
# Multilabel head-to-head (generic over the label set; used by the toxic
# benchmark). Defined here so the functions are picklable across the spawn
# boundary used by run_isolated.
# --------------------------------------------------------------------------- #
def read_multilabel(path: str):
    """Read JSONL into (texts, list-of-label-lists)."""
    import json

    texts, labels = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            texts.append(row["text"])
            labels.append(row.get("labels") or [])
    return texts, labels


def _ml_metrics(y_true, y_pred, y_score, label_names):
    import numpy as np
    from sklearn.metrics import f1_score, roc_auc_score

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    per = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_label = {label_names[i]: float(per[i]) for i in range(len(label_names))}
    try:
        roc = float(roc_auc_score(y_true, y_score, average="macro"))
    except ValueError:
        roc = None
    # subset accuracy (exact match of the whole label set)
    subset_acc = float(np.mean(np.all(np.asarray(y_true) == np.asarray(y_pred), axis=1)))
    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "roc_auc_macro": roc,
        "subset_accuracy": subset_acc,
        "per_label_f1": per_label,
    }


# Mirrored search spaces for the multilabel comparison.
ML_TW = dict(
    tokenizer="unicode",
    lowercase=True,
    ngram_range=[(1, 1), (1, 2)],
    n_features=[50_000, 200_000],
    n_hashes=2,
    feature_value="binary",
    epochs=(5, 15),
    learning_rate=(1e-2, 3e-1),
    l1=0.0,
    l2=(1e-7, 1e-3),
    batch_size=256,
    # Pinned off so the benchmark is a fair 0.5-vs-0.5 comparison with sklearn
    # (which also predicts at 0.5). tokenwiser tunes thresholds by *default* in
    # real use, which lifts its macro-F1 on imbalanced data.
    tune_thresholds=False,
)
ML_SK = {
    "vec__ngram_range": [(1, 1), (1, 2)],
    "vec__max_features": [50_000, 200_000],
    "vec__binary": [True, False],
    "clf__estimator__C": [0.1, 1.0, 10.0],
}


def multilabel_tokenwiser(*, train_path, test_path, trials, cv, seed,
                          threads=None, save_model=None):
    # rayon reads RAYON_NUM_THREADS when it first initializes its pool (during
    # train), so set it before importing/calling tokenwiser. threads=1 forces
    # the single-threaded baseline; None uses all cores.
    if threads is not None:
        os.environ["RAYON_NUM_THREADS"] = str(threads)

    import time

    import numpy as np
    import tokenwiser

    with PeakTreeRSS() as sampler:
        t0 = time.perf_counter()
        model, report = tokenwiser.train(
            train_path, text_field="text", labels_field="labels",
            trials=trials, cv=cv, seed=seed, **ML_TW,
        )
        total = time.perf_counter() - t0

        labels = model.labels
        idx = {l: i for i, l in enumerate(labels)}
        test_texts, test_gold = read_multilabel(test_path)
        p0 = time.perf_counter()
        proba = model.predict_proba(test_texts)
        predict_seconds = time.perf_counter() - p0

    y_true = np.zeros((len(test_gold), len(labels)), dtype=int)
    for r, labs in enumerate(test_gold):
        for l in labs:
            if l in idx:
                y_true[r, idx[l]] = 1
    y_score = np.array([[d.get(l, 0.0) for l in labels] for d in proba])
    y_pred = (y_score >= 0.5).astype(int)

    result = {
        "model": "tokenwiser",
        "variant": "1-thread" if threads == 1 else "parallel",
        "threads": threads if threads else os.cpu_count(),
        "n_train_rows": count_rows(train_path),
        "n_test_rows": len(test_gold),
        "n_labels": len(labels),
        "load_seconds": None,
        "fit_seconds": None,
        "total_seconds": total,
        "predict_seconds": predict_seconds,
        "best_cv_score": report["best_macro_f1"],
        "best_config": report["best_config"],
        "metrics": _ml_metrics(y_true, y_pred, y_score, labels),
        "model_bytes": None,
        # dense weight table footprint (this is where many labels show up)
        "weight_table_mb": len(labels) * report["best_config"]["n_features"] * 4 / 1e6,
        "vocab_terms": None,
        "peak_rss_mb": sampler.peak_mb,
    }
    if save_model:
        model.save(save_model)
        result["model_bytes"] = os.path.getsize(save_model)
    return result


def multilabel_sklearn(*, train_path, test_path, trials, cv, seed, n_jobs=1,
                       vectorizer="tfidf", save_model=None):
    import time

    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold, RandomizedSearchCV
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MultiLabelBinarizer

    with PeakTreeRSS() as sampler:
        l0 = time.perf_counter()
        texts, labels = read_multilabel(train_path)
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(labels)
        load_seconds = time.perf_counter() - l0

        # CountVectorizer drops IDF — a closer analog to tokenwiser's raw
        # binary/count features, isolating the IDF contribution to sklearn's
        # quality.
        vec = TfidfVectorizer() if vectorizer == "tfidf" else CountVectorizer()
        pipe = Pipeline([
            ("vec", vec),
            ("clf", OneVsRestClassifier(
                LogisticRegression(class_weight="balanced", max_iter=1000,
                                   solver="liblinear"))),
        ])
        search = RandomizedSearchCV(
            pipe, ML_SK, n_iter=trials, scoring="f1_macro",
            cv=KFold(n_splits=cv, shuffle=True, random_state=seed),
            random_state=seed, n_jobs=n_jobs, refit=True,
        )
        f0 = time.perf_counter()
        search.fit(texts, y)
        fit_seconds = time.perf_counter() - f0

        label_names = list(mlb.classes_)
        test_texts, test_gold = read_multilabel(test_path)
        y_true = mlb.transform(test_gold)
        p0 = time.perf_counter()
        y_score = search.predict_proba(test_texts)
        predict_seconds = time.perf_counter() - p0

    y_pred = (y_score >= 0.5).astype(int)
    fitted_vec = search.best_estimator_.named_steps["vec"]
    result = {
        "model": "sklearn",
        "variant": f"{vectorizer} n_jobs={n_jobs}",
        "vectorizer": vectorizer,
        "n_jobs": n_jobs if n_jobs != -1 else os.cpu_count(),
        "n_train_rows": count_rows(train_path),
        "n_test_rows": len(test_gold),
        "n_labels": len(label_names),
        "load_seconds": load_seconds,
        "fit_seconds": fit_seconds,
        "total_seconds": load_seconds + fit_seconds,
        "predict_seconds": predict_seconds,
        "best_cv_score": float(search.best_score_),
        "best_config": {k: str(v) for k, v in search.best_params_.items()},
        "metrics": _ml_metrics(y_true, y_pred, y_score, label_names),
        "model_bytes": None,
        "weight_table_mb": None,
        "vocab_terms": len(fitted_vec.vocabulary_),
        "peak_rss_mb": sampler.peak_mb,
    }
    if save_model:
        import pickle

        with open(save_model, "wb") as fh:
            pickle.dump(search.best_estimator_, fh)
        result["model_bytes"] = os.path.getsize(save_model)
    return result


# --------------------------------------------------------------------------- #
# Subprocess worker entrypoint. The notebook runs each benchmark variant via
# `python _benchlib.py <func_name> <json-kwargs>`, giving an isolated process
# per run (clean peak-RSS, including any child workers) without the
# multiprocessing-spawn-from-a-notebook pitfalls. kwargs are plain JSON scalars
# here (the tuple-bearing search spaces are module constants, not arguments).
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import json as _json

    func = globals()[sys.argv[1]]
    kwargs = _json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    sys.stdout.write(_json.dumps(func(**kwargs)))
