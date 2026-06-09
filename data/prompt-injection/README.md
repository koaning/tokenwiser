# Prompt Injection Benchmark Data

This directory contains the `core` configuration of
[`neuralchemy/Prompt-injection-dataset`](https://huggingface.co/datasets/neuralchemy/Prompt-injection-dataset),
pinned to revision `7d70432dfcf47a821612cbf9d34e9d9e3ad20e75`.

The included JSONL files are adapted to Tokenwiser's multilabel input:

- malicious rows use `"labels": ["malicious"]`
- benign rows use `"labels": []`

Regenerate the files with:

```bash
uv run --with pyarrow python scripts/download_prompt_injection.py
```

The source Parquet files are pinned and checksummed in `manifest.json`, but
removed after conversion to avoid storing the dataset twice. The source
dataset is licensed under Apache-2.0.
