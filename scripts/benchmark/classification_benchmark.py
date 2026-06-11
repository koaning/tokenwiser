# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.21.1",
#     "pandas>=2.0",
#     "matplotlib>=3.7",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd

    return Path, json, mo, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Classification benchmark

    A binary text-classification comparison between tokenwiser and a
    scikit-learn TF-IDF + logistic-regression pipeline. The example dataset is
    the prompt-injection `core` split, but the benchmark itself is generic.

    This notebook **visualizes** results produced by the two benchmark
    scripts — it does no training itself, so it stays fast and never imports
    tokenwiser or scikit-learn. Run the scripts first:

    ```bash
    python scripts/benchmark/benchmark_quality.py
    python scripts/benchmark/benchmark_scaling.py
    ```

    They write `benchmarks/quality.json` and `benchmarks/scaling.json`,
    which this notebook reads below.
    """)
    return


@app.cell
def _(Path, json):
    repo_root = Path(__file__).resolve().parents[2]
    bench_dir = repo_root / "benchmarks"

    def _load(name):
        path = bench_dir / name
        return json.loads(path.read_text()) if path.exists() else None

    quality = _load("quality.json")
    scaling = _load("scaling.json")
    return quality, scaling


@app.cell
def _(mo, quality):
    mo.stop(
        quality is None,
        mo.md("⚠️ `benchmarks/quality.json` not found — run "
              "`python scripts/benchmark/benchmark_quality.py` first."),
    )
    d = quality["dataset"]
    mo.md(
        f"""
        ## Head-to-head quality

        **Dataset:** `{d['name']}` (`{d['config']}`, rev `{d['revision'][:10]}`)
        **Run:** {quality['generated_at']} · {quality['platform']} · Python {quality['python']}
        **Search:** {quality['settings']['trials']} trials × {quality['settings']['cv']}-fold CV
        """
    )
    return


@app.cell
def _(mo, pd, quality):
    mo.stop(quality is None)

    def _quality_row(run):
        metrics = run["metrics"]
        return {
            "model": run["model"],
            "f1": round(metrics["f1"], 4),
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "roc_auc": round(metrics["roc_auc"], 4),
            "cv_f1": round(run["best_cv_score"], 4),
            "train_seconds": round(run["total_seconds"], 3),
            "predict_seconds": round(run["predict_seconds"], 4),
            "peak_rss_mb": round(run["peak_rss_mb"], 1),
            "model_kib": round(run["model_bytes"] / 1024, 1),
        }

    quality_table = pd.DataFrame([_quality_row(r) for r in quality["runs"]])
    mo.md(
        "`train_seconds` is measured from data-on-disk to a trained model for "
        "**both** models; everything runs single-threaded."
    )
    quality_table
    return


@app.cell
def _(json, mo, pd, quality):
    mo.stop(quality is None)
    best = pd.DataFrame(
        [
            {"model": r["model"], "best_config": json.dumps(r["best_config"])}
            for r in quality["runs"]
        ]
    )
    mo.md("### Selected configurations")
    best
    return


@app.cell
def _(mo, scaling):
    mo.stop(
        scaling is None,
        mo.md("⚠️ `benchmarks/scaling.json` not found — run "
              "`python scripts/benchmark/benchmark_scaling.py` first."),
    )
    mo.md(
        f"""
        ## Scaling with dataset size

        The base split ({scaling['dataset']['base_rows']:,} rows) is replicated
        up to each target size, then each model is trained once with a fixed
        config. We track wall-clock training time and **peak resident memory**
        (measured per run in an isolated process).
        """
    )
    return


@app.cell
def _(mo, pd, scaling):
    mo.stop(scaling is None)
    scaling_table = pd.DataFrame(
        [
            {
                "model": r["model"],
                "rows": r["n_train_rows"],
                "train_seconds": round(r["total_seconds"], 3),
                "peak_rss_mb": round(r["peak_rss_mb"], 1),
            }
            for r in scaling["runs"]
        ]
    ).sort_values(["model", "rows"])
    scaling_table
    return (scaling_table,)


@app.cell
def _(plt, scaling_table):
    fig_time, ax_time = plt.subplots(figsize=(7, 4))
    for _model, _grp in scaling_table.groupby("model"):
        ax_time.plot(_grp["rows"], _grp["train_seconds"], marker="o", label=_model)
    ax_time.set_xlabel("training rows")
    ax_time.set_ylabel("train seconds (disk → model)")
    ax_time.set_title("Training time vs dataset size")
    ax_time.legend()
    ax_time.grid(True, alpha=0.3)
    fig_time
    return


@app.cell
def _(plt, scaling_table):
    fig_mem, ax_mem = plt.subplots(figsize=(7, 4))
    for _model, _grp in scaling_table.groupby("model"):
        ax_mem.plot(_grp["rows"], _grp["peak_rss_mb"], marker="o", label=_model)
    ax_mem.set_xlabel("training rows")
    ax_mem.set_ylabel("peak RSS (MB)")
    ax_mem.set_title("Peak memory vs dataset size")
    ax_mem.legend()
    ax_mem.grid(True, alpha=0.3)
    fig_mem
    return


if __name__ == "__main__":
    app.run()
