from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_fingerprint.fingerprinting_dataset import _read_ground_truth_indices
from ai_fingerprint.result_collection import validate_collected_root
from ai_fingerprint.round_fingerprinting import (
    RoundInferenceConfig,
    build_round_dataset,
    discover_client_packet_sequences,
    evaluate_round_hierarchy,
    read_ground_truth_rounds,
)
from prepare_fingerprinting_dataset import discover_inputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Add proxy-only round-aware AI fingerprinting without replacing "
            "the existing complete-trace or real-time classifiers."
        )
    )
    parser.add_argument("--collected-root", default="collected_experiments")
    parser.add_argument("--dataset-output", default="fingerprinting_dataset/round_fingerprinting")
    parser.add_argument("--results-output", default="fingerprinting_results/round_fingerprinting")
    parser.add_argument("--bin-sec", type=float, default=0.25)
    parser.add_argument("--direction-dominance", type=float, default=0.65)
    parser.add_argument("--min-major-transfer-bytes", type=int, default=262144)
    parser.add_argument("--bridge-gap-sec", type=float, default=0.75)
    parser.add_argument("--max-round-sec", type=float, default=3600.0)
    parser.add_argument("--min-round-packets", type=int, default=20)
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Build the round dataset only; do not train/evaluate classifiers.",
    )
    args = parser.parse_args()

    collected_root = Path(args.collected_root).resolve()
    validation = validate_collected_root(collected_root)
    valid_run_ids = list(validation.get("valid_run_ids") or [])
    if not valid_run_ids:
        raise SystemExit("No VALID centrally collected runs are available.")

    print(f"VALID runs: {len(valid_run_ids)}")
    proxy_features, ground_truth, client_map, diagnostics = discover_inputs(
        collected_root,
        allowed_experiment_ids=set(valid_run_ids),
    )
    if not ground_truth:
        raise SystemExit("No matching ground-truth JSONL files were found.")
    if not client_map:
        raise SystemExit(
            "No confirmed proxy-connection -> federated-client mappings were found. "
            "Round fingerprinting intentionally refuses stale/unconfirmed connections."
        )

    print("Reconstructing exact client-facing packet sequences from proxy manifests...")
    client_packets = discover_client_packet_sequences(
        collected_root,
        valid_run_ids=valid_run_ids,
        client_map=client_map,
    )
    print(f"Resolved client packet traces: {len(client_packets)}")
    if not client_packets:
        raise SystemExit(
            "No raw proxy packet-sequence CSVs were found in the central copies. "
            "The round detector needs packet timestamps and directions. Raw PCAP is not required."
        )

    _experiment_labels, client_labels = _read_ground_truth_indices(ground_truth)
    gt_rounds = read_ground_truth_rounds(ground_truth)

    cfg = RoundInferenceConfig(
        bin_sec=args.bin_sec,
        direction_dominance=args.direction_dominance,
        min_major_transfer_bytes=args.min_major_transfer_bytes,
        bridge_gap_sec=args.bridge_gap_sec,
        max_round_sec=args.max_round_sec,
        min_round_packets=args.min_round_packets,
    )

    print("Inferring round boundaries from proxy-observable traffic only...")
    dataset = build_round_dataset(
        client_packets=client_packets,
        client_labels=client_labels,
        ground_truth_rounds=gt_rounds,
        output_dir=Path(args.dataset_output),
        inference_config=cfg,
    )
    print(json.dumps(dataset, indent=2))

    if not args.skip_evaluation:
        print("\nEvaluating round-level and round-fusion hierarchical classifiers...")
        result = evaluate_round_hierarchy(
            x_csv=dataset["x_csv"],
            y_csv=dataset["y_csv"],
            output_root=Path(args.results_output),
        )
        print(json.dumps(result, indent=2))

    print("\nExisting complete-trace and real-time models were not modified.")
    print("Round number and endpoint ground-truth boundaries are metadata only, never predictors.")


if __name__ == "__main__":
    main()
