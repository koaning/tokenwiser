"""Download the pinned prompt-injection benchmark dataset from Hugging Face."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq


REVISION = "7d70432dfcf47a821612cbf9d34e9d9e3ad20e75"
BASE_URL = (
    "https://huggingface.co/datasets/neuralchemy/"
    f"Prompt-injection-dataset/resolve/{REVISION}/core"
)
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "prompt-injection" / "core"
SPLITS = ("train", "validation", "test")


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return

    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    temporary.replace(destination)


def convert_for_tokenwiser(source: Path, destination: Path) -> None:
    table = pq.read_table(source, columns=["text", "label"])
    texts = table.column("text").to_pylist()
    labels = table.column("label").to_pylist()
    with destination.open("w", encoding="utf-8") as output:
        for text, label in zip(texts, labels):
            row = {
                "text": text,
                "labels": ["malicious"] if label == 1 else [],
            }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest = {
        "dataset": "neuralchemy/Prompt-injection-dataset",
        "config": "core",
        "revision": REVISION,
        "files": {},
    }
    for split in SPLITS:
        parquet_name = f"{split}-00000-of-00001.parquet"
        parquet_path = OUTPUT_DIR / parquet_name
        jsonl_path = OUTPUT_DIR / f"{split}.jsonl"
        download(f"{BASE_URL}/{parquet_name}", parquet_path)
        row_count = pq.read_metadata(parquet_path).num_rows
        checksum = sha256(parquet_path)
        convert_for_tokenwiser(parquet_path, jsonl_path)
        manifest["files"][parquet_name] = {
            "sha256": checksum,
            "rows": row_count,
        }
        parquet_path.unlink()

    manifest_path = OUTPUT_DIR.parent / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Dataset ready in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
