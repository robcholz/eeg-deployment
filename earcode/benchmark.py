import argparse
import csv
import time
from pathlib import Path

import numpy as np

import config as cfg
from main import run_pipeline


def benchmark(model_names):
    results = []
    out_dir = Path(__file__).resolve().parent / "benchmark_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in model_names:
        cfg.model_name = name
        start = time.perf_counter()
        res = run_pipeline(model_name=name)
        duration = time.perf_counter() - start

        mean = float(res.mean().item())
        std = float(res.std().item())

        np.savetxt(out_dir / f"{name}_result.csv", res.numpy(), delimiter=",")
        results.append((name, mean, std, duration))
        print(f"{name}: {mean:.2f}% (SD: {std:.2f}%), runtime {duration:.1f}s")

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "accuracy_mean", "accuracy_std", "runtime_seconds"])
        for row in results:
            writer.writerow(row)

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run ASAD benchmarks.")
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(cfg.model_names[:2]),
        help="Comma-separated list of model names."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all models in config."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.all:
        selected = cfg.model_names
    else:
        selected = [m.strip() for m in args.models.split(",") if m.strip()]

    benchmark(selected)

