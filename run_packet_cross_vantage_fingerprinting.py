from __future__ import annotations

"""Single-command, non-destructive packet-budget orchestration.

Existing packet-budget CSVs are reused by default. Pass --rebuild only when
you intentionally want to regenerate them.
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "fingerprinting_dataset" / "packet_cross_vantage"
RESULT_ROOT = ROOT / "fingerprinting_results" / "packet_cross_vantage"


def _k1_coverage():
    inventory = DATASET_ROOT / "packet_dataset_inventory.csv"
    k1 = {"client": 0, "server": 0, "proxy": 0}

    if inventory.exists():
        with inventory.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    budget = int(row["packet_budget"])
                except (KeyError, TypeError, ValueError):
                    continue
                if budget != 1:
                    continue
                role = str(row.get("source_role", ""))
                if role in k1:
                    try:
                        k1[role] = int(row.get("samples", 0) or 0)
                    except (TypeError, ValueError):
                        pass
    return k1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Regenerate packet-budget datasets. Default: reuse existing CSVs.",
    )
    parser.add_argument(
        "--packet-counts",
        default="1,5,10,25,50,100,250,500",
    )
    parser.add_argument("--max-samples-per-trace", type=int, default=5000)
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-folds", type=int, default=5)
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument(
        "--packet-fusion",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    builder = ROOT / "build_packet_cross_vantage_dataset.py"
    trainer = ROOT / "train_packet_cross_vantage.py"

    if not trainer.exists():
        raise SystemExit("train_packet_cross_vantage.py is missing from project root.")

    requested = [
        int(value)
        for value in args.packet_counts.split(",")
        if value.strip()
    ]
    missing = [
        k
        for k in requested
        if not (DATASET_ROOT / f"packet_{k}.csv").exists()
    ]

    need_build = args.rebuild or not DATASET_ROOT.exists() or bool(missing)

    if need_build:
        if not builder.exists():
            raise SystemExit("Packet dataset builder is missing.")
        if missing and not args.rebuild:
            print(f"[dataset] Missing requested packet budgets {missing}; building dataset.")
        else:
            print("[dataset] Rebuilding packet-budget datasets...")
        subprocess.run(
            [sys.executable, str(builder)],
            cwd=str(ROOT),
            check=True,
        )
    else:
        print(f"[dataset] Reusing existing packet-budget CSVs in {DATASET_ROOT}")

    k1 = _k1_coverage()
    print(
        "\nK=1 inventory: "
        f"client={k1['client']:,} "
        f"server={k1['server']:,} "
        f"proxy={k1['proxy']:,}"
    )

    if k1["proxy"] == 0 and not (DATASET_ROOT / "packet_1.csv").exists():
        raise SystemExit("No proxy K=1 packet dataset exists.")

    if k1["client"] == 0 and k1["server"] == 0:
        print(
            "\n[coverage] Historical endpoint packet sequences are absent. "
            "Running proxy->proxy experiment-disjoint packet fingerprinting only. "
            "This is NOT labelled endpoint->proxy."
        )
        modes = "proxy_to_proxy"
    else:
        modes = "client_to_proxy,server_to_proxy,client_server_to_proxy,proxy_to_proxy"

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

    command.append("--packet-fusion" if args.packet_fusion else "--no-packet-fusion")

    print("\n[evaluate] Starting memory-safe experiment-disjoint evaluation...")
    subprocess.run(command, cwd=str(ROOT), check=True)

    print("\nComplete:", RESULT_ROOT)


if __name__ == "__main__":
    main()
