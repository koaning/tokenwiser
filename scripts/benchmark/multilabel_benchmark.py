# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.21.1",
#     "pandas>=2.0",
#     "matplotlib>=3.7",
# ]
# ///
#
# NOTE: run this with the *project* environment (where tokenwiser is built and
# scikit-learn/psutil are installed), not an isolated uv env — the notebook
# shells out to `sys.executable` to run each benchmark variant:
#
#     .venv/bin/marimo edit scripts/benchmark/multilabel_benchmark.py

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import os
    import random
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd

    BENCHLIB = str(Path(__file__).resolve().parent / "_benchlib.py")
    DEFAULT_DATA = str(Path(__file__).resolve().parents[2] / "data" / "toxic" / "toxic.jsonl")
    return (
        BENCHLIB,
        DEFAULT_DATA,
        json,
        mo,
        os,
        pd,
        plt,
        random,
        subprocess,
        sys,
        tempfile,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Multilabel benchmark — runnable

    Runs the benchmark itself (no precomputed JSON): tokenwiser vs
    scikit-learn TF-IDF and CountVectorizer pipelines, on any multilabel
    JSONL (`{"text": ..., "labels": [...]}`). Each variant runs in its own
    subprocess for a clean peak-RSS reading (parent + worker processes).

    Variants:
    - **tokenwiser (parallel)** — rayon across all cores
    - **tokenwiser (1-thread)** — single-core baseline
    - **sklearn TF-IDF (parallel)** — IDF-weighted features
    - **sklearn Count (parallel)** — raw counts, no IDF (closest analog to
      tokenwiser's features — isolates what IDF buys)

    All predict at a fixed **0.5** threshold (no per-label tuning on either
    side), so F1 is like-for-like.
    """)
    return


@app.cell
def _(DEFAULT_DATA, mo):
    dataset = mo.ui.text(value=DEFAULT_DATA, label="dataset (JSONL)", full_width=True)
    max_rows = mo.ui.dropdown(
        options={"5,000": 5000, "20,000": 20000, "50,000": 50000, "full": -1},
        value="20,000", label="rows (subsample for speed)",
    )
    trials = mo.ui.slider(1, 12, value=4, label="trials")
    cv = mo.ui.slider(2, 5, value=3, label="cv folds")
    run = mo.ui.run_button(label="Run benchmark")
    mo.vstack([dataset, mo.hstack([max_rows, trials, cv]), run])
    return cv, dataset, max_rows, run, trials


@app.cell
def _(
    BENCHLIB,
    cv,
    dataset,
    json,
    max_rows,
    mo,
    os,
    random,
    run,
    subprocess,
    sys,
    tempfile,
    trials,
):
    mo.stop(not run.value, mo.md("Set options above, then click **Run benchmark**."))
    mo.stop(not os.path.exists(dataset.value),
            mo.md(f"⚠️ dataset not found: `{dataset.value}`"))

    # Load, optionally subsample, and split 80/20 (seeded) into temp files.
    rows = []
    with open(dataset.value, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                labs = r.get("labels") or []
                rows.append((r["text"], [labs] if isinstance(labs, str) else labs))
    idx = list(range(len(rows)))
    random.Random(42).shuffle(idx)
    if max_rows.value and max_rows.value > 0:
        idx = idx[: max_rows.value]
    cut = int(len(idx) * 0.8)
    tmp = tempfile.mkdtemp()

    def _write(ids, path):
        with open(path, "w", encoding="utf-8") as fh:
            for i in ids:
                text, labs = rows[i]
                fh.write(json.dumps({"text": text, "labels": labs}) + "\n")

    train_path = os.path.join(tmp, "train.jsonl")
    test_path = os.path.join(tmp, "test.jsonl")
    _write(idx[:cut], train_path)
    _write(idx[cut:], test_path)

    def run_variant(name, func, **kw):
        kw = dict(train_path=train_path, test_path=test_path,
                  trials=trials.value, cv=cv.value, seed=42, **kw)
        out = subprocess.run([sys.executable, BENCHLIB, func, json.dumps(kw)],
                             capture_output=True, text=True)
        if out.returncode != 0:
            raise RuntimeError(f"{name} failed:\n{out.stderr[-2000:]}")
        return dict(json.loads(out.stdout), name=name)

    plan = [
        ("tokenwiser (parallel)", "multilabel_tokenwiser", dict(threads=None)),
        ("tokenwiser (1-thread)", "multilabel_tokenwiser", dict(threads=1)),
        ("sklearn TF-IDF (parallel)", "multilabel_sklearn", dict(n_jobs=-1, vectorizer="tfidf")),
        ("sklearn Count (parallel)", "multilabel_sklearn", dict(n_jobs=-1, vectorizer="count")),
    ]
    results = []
    with mo.status.spinner(title="Running variants…") as _sp:
        for _name, _func, _kw in plan:
            _sp.update(subtitle=_name)
            results.append(run_variant(_name, _func, **_kw))
    n_train, n_test = cut, len(idx) - cut
    results
    return n_test, n_train, results


@app.cell
def _(n_test, n_train, pd, results):
    summary = pd.DataFrame([
        {
            "variant": r["name"],
            "train_s": round(r["total_seconds"], 1),
            "predict_s": round(r["predict_seconds"], 2),
            "peak_RSS_MB": round(r["peak_rss_mb"]),
            "model_MB": round(r["model_bytes"] / 1e6, 1) if r["model_bytes"] else None,
            "macro_F1": round(r["metrics"]["macro_f1"], 4),
            "micro_F1": round(r["metrics"]["micro_f1"], 4),
            "roc_auc": round(r["metrics"]["roc_auc_macro"], 4),
            "vocab/weights": (f"{r['vocab_terms']:,} terms" if r.get("vocab_terms")
                              else f"{r['weight_table_mb']:.1f} MB weights"),
        }
        for r in results
    ])
    summary.attrs["caption"] = f"train {n_train:,} / test {n_test:,}"
    summary
    return


@app.cell
def _(plt, results):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    names = [r["name"] for r in results]
    colors = ["#2a9d8f", "#8ecae6", "#e9c46a", "#e76f51"][: len(results)]
    axes[0].bar(names, [r["total_seconds"] for r in results], color=colors)
    axes[0].set_title("train time (s)")
    axes[1].bar(names, [r["peak_rss_mb"] for r in results], color=colors)
    axes[1].set_title("peak RSS (MB) — process tree")
    axes[2].bar(names, [(r["model_bytes"] or 0) / 1e6 for r in results], color=colors)
    axes[2].set_title("model size (MB)")
    for _ax in axes:
        _ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig
    return


@app.cell
def _(plt, results):
    fig_q, axes_q = plt.subplots(1, 2, figsize=(11, 3.8))
    names_q = [r["name"] for r in results]
    colors_q = ["#2a9d8f", "#8ecae6", "#e9c46a", "#e76f51"][: len(results)]
    axes_q[0].bar(names_q, [r["metrics"]["macro_f1"] for r in results], color=colors_q)
    axes_q[0].set_title("macro-F1 @ 0.5")
    axes_q[0].set_ylim(0, 1)
    axes_q[1].bar(names_q, [r["metrics"]["roc_auc_macro"] for r in results], color=colors_q)
    axes_q[1].set_title("ROC-AUC (macro)")
    axes_q[1].set_ylim(0.85, 1.0)
    for _ax in axes_q:
        _ax.tick_params(axis="x", rotation=30)
    fig_q.tight_layout()
    fig_q
    return


@app.cell
def _(mo, pd, plt, results):
    # Per-label F1: tokenwiser (parallel) vs each sklearn vectorizer.
    chosen = [r for r in results
              if r["name"] in ("tokenwiser (parallel)", "sklearn TF-IDF (parallel)",
                               "sklearn Count (parallel)")]
    labels = sorted(chosen[0]["metrics"]["per_label_f1"])
    df = pd.DataFrame({r["name"]: [r["metrics"]["per_label_f1"][l] for l in labels]
                       for r in chosen}, index=labels)

    fig_pl, ax_pl = plt.subplots(figsize=(10, 4))
    width = 0.8 / len(chosen)
    for _k, _col in enumerate(df.columns):
        ax_pl.bar([i + (_k - (len(chosen) - 1) / 2) * width for i in range(len(labels))],
                  df[_col], width=width, label=_col)
    ax_pl.set_xticks(range(len(labels)))
    ax_pl.set_xticklabels(labels, rotation=25, ha="right")
    ax_pl.set_ylabel("F1 @ 0.5")
    ax_pl.set_title("Per-label F1")
    ax_pl.legend(fontsize=8)
    fig_pl.tight_layout()
    mo.vstack([fig_pl, df.round(3)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Reading this

    - **tokenwiser parallel vs 1-thread** shows what rayon buys (results are
      identical — same seed → same model regardless of thread count).
    - **TF-IDF vs Count** isolates IDF: Count drops it, so the TF-IDF − Count
      F1 gap is roughly what IDF contributes — and it's the part tokenwiser
      (no IDF) is missing.
    - **macro-F1 vs ROC-AUC:** tokenwiser ranks competitively (ROC-AUC) but
      its fixed-0.5 macro-F1 trails, because balanced weighting inflates
      probabilities; the gap concentrates in rare labels. Both sides use 0.5
      for a fair comparison.
    """)
    return


if __name__ == "__main__":
    app.run()
