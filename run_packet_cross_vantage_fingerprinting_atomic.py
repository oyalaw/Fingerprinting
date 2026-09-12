from __future__ import annotations

"""Single-command packet fingerprinting runner using the atomic sampled builder."""

import argparse
import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASET_ROOT = (
    ROOT
    / "fingerprinting_dataset"
    / "packet_cross_vantage"
)
RESULT_ROOT = (
    ROOT
    / "fingerprinting_results"
    / "packet_cross_vantage"
)


def inventory_coverage():
    inventory = (
        DATASET_ROOT
        / "packet_dataset_inventory.csv"
    )
    result = {}
    if not inventory.exists():
        return result

    with inventory.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        for row in csv.DictReader(handle):
            key = (
                int(row["packet_budget"]),
                str(row["source_role"]),
            )
            result[key] = {
                "samples": int(
                    row.get("samples", 0) or 0
                ),
                "available_samples": int(
                    row.get(
                        "available_samples",
                        row.get("samples", 0),
                    )
                    or 0
                ),
                "client_traces": int(
                    row.get("client_traces", 0) or 0
                ),
                "experiments": int(
                    row.get("experiments", 0) or 0
                ),
            }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help=(
            "Build a fresh validated packet dataset using "
            "the atomic trace-balanced builder."
        ),
    )
    parser.add_argument(
        "--packet-counts",
        default="1,5,10,25,50,100,250,500",
    )
    parser.add_argument(
        "--max-samples-per-trace",
        type=int,
        default=5000,
        help=(
            "Evaluation cap per trace. Default 5000."
        ),
    )
    parser.add_argument(
        "--dataset-samples-per-trace",
        type=int,
        default=5000,
        help=(
            "Materialization cap used only when rebuilding "
            "the compact packet dataset. Default 5000."
        ),
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=500_000,
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--packet-fusion",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    budgets = [
        int(x)
        for x in args.packet_counts.split(",")
        if x.strip()
    ]

    builder = (
        ROOT
        / "build_packet_cross_vantage_atomic_sampled.py"
    )
    trainer = (
        ROOT
        / "train_packet_cross_vantage.py"
    )

    if not trainer.exists():
        raise SystemExit(
            "train_packet_cross_vantage.py is missing."
        )

    missing = [
        k
        for k in budgets
        if not (
            DATASET_ROOT / f"packet_{k}.csv"
        ).exists()
    ]

    if args.rebuild or missing:
        if not builder.exists():
            raise SystemExit(
                "build_packet_cross_vantage_atomic_sampled.py "
                "is missing."
            )

        print(
            "[dataset] Building a validated compact packet "
            "dataset atomically..."
        )
        subprocess.run(
            [
                sys.executable,
                str(builder),
                "--packet-counts",
                args.packet_counts,
                "--max-samples-per-trace",
                str(args.dataset_samples_per_trace),
                "--sampling-seed",
                str(args.sampling_seed),
            ],
            cwd=str(ROOT),
            check=True,
        )
    else:
        print(
            "[dataset] Reusing existing packet-budget "
            f"CSVs in {DATASET_ROOT}"
        )

    coverage = inventory_coverage()
    k1 = coverage.get(
        (1, "proxy"),
        {
            "samples": 0,
            "available_samples": 0,
            "client_traces": 0,
            "experiments": 0,
        },
    )

    print(
        "\nK=1 proxy coverage: "
        f"written_samples={k1['samples']:,} "
        f"available_samples={k1['available_samples']:,} "
        f"traces={k1['client_traces']:,} "
        f"experiments={k1['experiments']:,}"
    )

    # Current historical corpus is expected to have endpoint packet sequences
    # absent. Infer the usable mode from inventory rather than fabrication.
    client_samples = coverage.get(
        (1, "client"),
        {},
    ).get("samples", 0)
    server_samples = coverage.get(
        (1, "server"),
        {},
    ).get("samples", 0)

    if client_samples == 0 and server_samples == 0:
        print(
            "\n[coverage] Historical endpoint packet "
            "sequences are absent. Running proxy->proxy "
            "experiment-disjoint packet fingerprinting only."
        )
        modes = "proxy_to_proxy"
    else:
        modes = (
            "client_to_proxy,"
            "server_to_proxy,"
            "client_server_to_proxy,"
            "proxy_to_proxy"
        )

    command = [
        sys.executable,
        str(trainer),
        "--vantage-modes",
        modes,
        "--packet-counts",
        args.packet_counts,
        "--max-samples-per-trace",
        str(args.max_samples_per_trace),
        "--chunksize",
        str(args.chunksize),
        "--n-estimators",
        str(args.n_estimators),
        "--top-k",
        str(args.top_k),
        "--max-folds",
        str(args.max_folds),
        "--sampling-seed",
        str(args.sampling_seed),
    ]
    command.append(
        "--packet-fusion"
        if args.packet_fusion
        else "--no-packet-fusion"
    )

    print(
        "\n[evaluate] Starting experiment-disjoint "
        "packet evaluation..."
    )
    subprocess.run(
        command,
        cwd=str(ROOT),
        check=True,
    )

    print(
        "\nComplete:",
        RESULT_ROOT,
    )


if __name__ == "__main__":
    main()
