from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ai_fingerprint.fingerprinting_dataset import (
    FingerprintingDataError,
    build_fingerprinting_dataset,
)
from ai_fingerprint.result_collection import validate_collected_root
from prepare_fingerprinting_dataset import discover_inputs


def main() -> None:
    project_root = Path(__file__).resolve().parent
    collected_root = project_root / "collected_experiments"
    validation = validate_collected_root(collected_root)
    validation_path = collected_root / "collection_validation.json"
    collected_root.mkdir(parents=True, exist_ok=True)
    validation_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("Central collection validation:")
    for run in validation["runs"]:
        print(
            f"  {run['run_id']}: {run['status']} "
            f"participants={','.join(run['participants'])}"
        )
        for issue in run["issues"]:
            print(f"    - {issue}")

    valid_run_ids = validation["valid_run_ids"]
    if not valid_run_ids:
        raise SystemExit(
            "No VALID collected runs are available. Start/verify the collector "
            "and retry pending participant uploads first."
        )

    (
        proxy_features,
        ground_truth,
        client_map,
        diagnostics,
    ) = discover_inputs(
        collected_root,
        allowed_experiment_ids=set(valid_run_ids),
    )

    if not proxy_features:
        raise SystemExit(
            "VALID runs exist, but no proxy *_features.csv files were found. "
            "Ensure the proxy completed post-capture extraction before upload."
        )
    if not ground_truth:
        raise SystemExit(
            "VALID runs exist, but no matching client/server ground-truth JSONL "
            "files were found."
        )

    print("\nFingerprinting inputs from VALID central copies only:")
    print(f"  runs: {', '.join(valid_run_ids)}")
    print(f"  proxy feature files: {len(proxy_features)}")
    print(f"  ground-truth files: {len(ground_truth)}")
    print(f"  resolved client mappings: {len(client_map)}")

    output_dir = project_root / "fingerprinting_dataset"
    try:
        result = build_fingerprinting_dataset(
            proxy_feature_csvs=proxy_features,
            ground_truth_jsonls=ground_truth,
            output_dir=output_dir,
            prefix="fingerprinting",
            client_capture_id_map=client_map,
        )
    except FingerprintingDataError as exc:
        raise SystemExit(f"Central fingerprinting dataset build failed: {exc}") from exc

    central_summary = {
        "collection_validation": str(validation_path),
        "valid_run_ids": valid_run_ids,
        "excluded_runs": [
            run for run in validation["runs"] if run["status"] != "VALID"
        ],
        "proxy_feature_files": [str(path) for path in proxy_features],
        "ground_truth_files": [str(path) for path in ground_truth],
        "resolved_client_mappings": [
            {
                "run_id": run_id,
                "client_capture_id": capture_id,
                "client_id": client_id,
            }
            for (run_id, capture_id), client_id in sorted(client_map.items())
        ],
        "dataset": result,
        "predictor_policy": (
            "Only proxy-observable network features enter X. Client/server "
            "labels, client IDs, system telemetry, OS/device metadata, IPs, "
            "ports, and experiment IDs are excluded from predictor columns."
        ),
    }
    summary_path = output_dir / "central_fingerprinting_inputs.json"
    summary_path.write_text(
        json.dumps(central_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"  dataset: {result}")
    print(f"  input audit: {summary_path}")

    print("\nTraining hierarchical fingerprint models...")
    subprocess.run(
        [sys.executable, str(project_root / "train_fingerprinting_models.py")],
        cwd=str(project_root),
        check=True,
    )
    print("\nCentral fingerprinting workflow complete.")


if __name__ == "__main__":
    main()
