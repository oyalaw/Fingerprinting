from __future__ import annotations

"""Quick coverage validator for packet_cross_vantage datasets."""

import argparse
import pandas as pd
from pathlib import Path


def audit(path: Path, chunksize: int = 500_000):
    families = set()
    architectures = set()
    variants = set()
    experiments = set()
    traces = set()
    rows = 0

    for chunk in pd.read_csv(
        path,
        usecols=[
            "experiment_id",
            "client_id",
            "source_role",
            "family",
            "architecture",
            "variant",
        ],
        chunksize=chunksize,
    ):
        rows += len(chunk)
        families.update(
            chunk["family"].dropna().astype(str)
        )
        architectures.update(
            chunk["architecture"].dropna().astype(str)
        )
        variants.update(
            chunk["variant"].dropna().astype(str)
        )
        experiments.update(
            chunk["experiment_id"].dropna().astype(str)
        )
        traces.update(
            zip(
                chunk["experiment_id"].astype(str),
                chunk["client_id"].astype(str),
                chunk["source_role"].astype(str),
            )
        )

    print(f"\n=== {path.name} ===")
    print(f"rows          : {rows:,}")
    print(f"experiments   : {len(experiments)}")
    print(f"client traces : {len(traces)}")
    print(f"families      : {sorted(families)}")
    print(f"architectures : {sorted(architectures)}")
    print(f"variants      : {sorted(variants)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        default="fingerprinting_dataset/packet_cross_vantage",
    )
    parser.add_argument(
        "--packet-counts",
        default="1,100",
    )
    args = parser.parse_args()

    root = Path(args.dataset_root)
    for value in args.packet_counts.split(","):
        value = value.strip()
        if not value:
            continue
        path = root / f"packet_{int(value)}.csv"
        if not path.exists():
            print(f"[missing] {path}")
            continue
        audit(path)


if __name__ == "__main__":
    main()
