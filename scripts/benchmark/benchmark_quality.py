"""Head-to-head quality/speed/size benchmark on the prompt-injection `core` split.

Trains tokenwiser and a scikit-learn TF-IDF + logistic-regression pipeline with
mirrored randomized search, evaluates both once on the untouched test split, and
writes machine-readable results to ``benchmarks/quality.json``.

Both runs are timed from data-on-disk to a trained model and run single-threaded
(see ``_benchlib`` for the fairness rationale).

    python scripts/benchmark/benchmark_quality.py --trials 12 --cv 3
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _benchlib as bench  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "prompt-injection",
        help="directory containing core/{train,test}.jsonl and manifest.json",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "quality.json")
    args = parser.parse_args()

    core = args.data_dir / "core"
    train_path = str(core / "train.jsonl")
    test_path = str(core / "test.jsonl")
    manifest = json.loads((args.data_dir / "manifest.json").read_text())

    with tempfile.TemporaryDirectory() as tmp:
        tw = bench.run_isolated(
            bench.train_tokenwiser,
            train_path=train_path,
            test_path=test_path,
            trials=args.trials,
            cv=args.cv,
            seed=args.seed,
            params=bench.QUALITY_TW,
            save_model=str(Path(tmp) / "model.tw"),
        )
        print(f"tokenwiser: {tw['total_seconds']:.2f}s, f1={tw['metrics']['f1']:.4f}")

        sk = bench.run_isolated(
            bench.train_sklearn,
            train_path=train_path,
            test_path=test_path,
            trials=args.trials,
            cv=args.cv,
            seed=args.seed,
            space=bench.QUALITY_SK,
            save_model=str(Path(tmp) / "model.pkl"),
        )
        print(f"sklearn:    {sk['total_seconds']:.2f}s, f1={sk['metrics']['f1']:.4f}")

    results = {
        "benchmark": "quality",
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "dataset": {
            "name": manifest["dataset"],
            "config": manifest["config"],
            "revision": manifest["revision"],
        },
        "settings": {"trials": args.trials, "cv": args.cv, "seed": args.seed},
        "runs": [tw, sk],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
