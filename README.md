<img src="docs/logo-tokw.png" width=280 align="right">

# tokenwiser

> Bag of, not words, but tricks!

`tokenwiser` is a small, fast **multilabel text classifier** with a Rust core
(via [PyO3](https://pyo3.rs)) and a tiny Python API. It streams JSONL, hashes
text into Bloom features, trains one sparse logistic regression per label, and
runs a cross-validated randomized hyperparameter search — all from a single
`train(...)` call.

## Install (from source)

You need a Rust toolchain and [maturin](https://www.maturin.rs):

```bash
python -m venv .venv && source .venv/bin/activate
pip install maturin
maturin develop --release
```

## Quickstart

Training data is JSONL with a text field and a labels field:

```json
{"text": "great wonderful movie", "labels": ["pos"]}
{"text": "awful boring film", "labels": ["neg"]}
```

```python
import tokenwiser

model, report = tokenwiser.train(
    "train.jsonl",
    text_field="text",
    labels_field="labels",
    trials=12,
    cv=3,
    seed=42,
    tokenizer=["unicode", "whitespace"],
    lowercase=True,
    ngram_range=[(1, 1), (1, 2)],
    n_features=(10_000, 100_000),
    n_hashes=(1, 4),
    feature_value=["binary", "count"],
    epochs=(2, 10),
    learning_rate=(1e-4, 1e-1),
    l1=0.0,
    l2=(1e-8, 1e-2),
    batch_size=256,
)

labels = model.predict(["some text"])
probabilities = model.predict_proba(["some text"])
model.save("model.tw")

model = tokenwiser.load("model.tw")

report.to_dict()
report.save("report.json")
```

## Search-space semantics

Every searchable argument follows one rule set:

| You pass        | Meaning                              | Example |
|-----------------|--------------------------------------|---------|
| **scalar**      | fixed value                          | `epochs=5` |
| **list**        | discrete choices                     | `tokenizer=["unicode", "whitespace"]` |
| **tuple**       | randomized range (parameter-aware)   | `learning_rate=(1e-4, 1e-1)` |

Details:

- Integer ranges (`epochs`, `n_features`, `n_hashes`) are **inclusive** and uniform.
- `learning_rate`, `l1`, and `l2` ranges are **log-uniform** (so they cannot include zero — use a fixed value or a list for that).
- `ngram_range=(1, 2)` is a **fixed** structured tuple; `[(1, 1), (1, 2)]` is a choice.
- `batch_size` is a fixed execution setting, never searched.
- Class weighting is always **balanced**, the search optimizes **cross-validated macro F1**, and prediction uses a fixed **0.5** threshold.

There is no CLI in v1.

## Benchmark

A text-classification comparison against a scikit-learn TF-IDF + logistic
regression baseline, using the pinned `core` split of
[`neuralchemy/Prompt-injection-dataset`](https://huggingface.co/datasets/neuralchemy/Prompt-injection-dataset).
Two scripts produce machine-readable results, and one marimo notebook
visualizes them:

```bash
# 1. Fetch the pinned dataset
uv run --with pyarrow python scripts/download_prompt_injection.py

# 2. Run the benchmarks (write benchmarks/{quality,scaling}.json)
uv run --with 'scikit-learn>=1.4' python scripts/benchmark/benchmark_quality.py
uv run --with 'scikit-learn>=1.4' python scripts/benchmark/benchmark_scaling.py

# 3. Visualize (reads the JSON; no training, no sklearn needed)
uv run --with 'marimo>=0.21.1' --with 'pandas>=2.0' --with 'matplotlib>=3.7' \
  marimo edit scripts/benchmark/classification_benchmark.py
```

Both benchmarks time each model **from data-on-disk to a trained model** and run
single-threaded, so the comparison is apples-to-apples. The two scripts live in
`scripts/benchmark/`: `benchmark_quality.py` compares quality/speed/size on the
real split; `benchmark_scaling.py` replicates the data to growing sizes and
tracks training time and peak memory.

## Development

```bash
make develop   # build the extension into the current env
make test      # run the Python test suite
make check     # build + test
```
