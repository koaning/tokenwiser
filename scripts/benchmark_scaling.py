"""Scaling sweep: how train time and peak memory grow with dataset size.

The base `core` training split is replicated (rows cycled) up to a series of
target sizes. For each size, tokenwiser and the scikit-learn pipeline are each
trained once — in an isolated child process — with a *fixed* config (one CV
pass, no real search), so the numbers reflect training cost rather than search
budget. Results are written to ``benchmarks/scaling.json``.

    python scripts/benchmark_scaling.py --sizes 5000 25000 100000
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import platform
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _benchlib as bench  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIZES = [5_000, 10_000, 25_000, 50_000, 100_000, 200_000]


def make_scaled_file(base_lines, n_rows, destination) -> None:
    """Write ``n_rows`` rows to ``destination`` by cycling the base rows."""
    with open(destination, "w", encoding="utf-8") as out:
        for line in itertools.islice(itertools.cycle(base_lines), n_rows):
            out.write(line if line.endswith("\n") else line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--cv", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "prompt-injection",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "scaling.json")
    args = parser.parse_args()

    base_path = args.data_dir / "core" / "train.jsonl"
    base_lines = [ln for ln in base_path.read_text().splitlines() if ln.strip()]
    manifest = json.loads((args.data_dir / "manifest.json").read_text())

    runs = []
    with tempfile.TemporaryDirectory() as tmp:
        for n_rows in args.sizes:
            scaled = Path(tmp) / f"train_{n_rows}.jsonl"
            make_scaled_file(base_lines, n_rows, scaled)

            tw = bench.run_isolated(
                bench.train_tokenwiser,
                train_path=str(scaled),
                trials=args.trials,
                cv=args.cv,
                seed=args.seed,
                params=bench.SCALING_TW,
            )
            sk = bench.run_isolated(
                bench.train_sklearn,
                train_path=str(scaled),
                trials=args.trials,
                cv=args.cv,
                seed=args.seed,
                space=bench.SCALING_SK,
            )
            for r in (tw, sk):
                print(
                    f"{r['model']:>10}  rows={n_rows:>7,}  "
                    f"{r['total_seconds']:7.2f}s  {r['peak_rss_mb']:7.1f} MB"
                )
                runs.append(r)

    results = {
        "benchmark": "scaling",
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "dataset": {
            "name": manifest["dataset"],
            "config": manifest["config"],
            "revision": manifest["revision"],
            "base_rows": len(base_lines),
        },
        "settings": {"trials": args.trials, "cv": args.cv, "seed": args.seed},
        "sizes": args.sizes,
        "runs": runs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
