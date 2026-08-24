from __future__ import annotations

from pathlib import Path
import csv

from ai_fingerprint.fingerprinting_dataset import (
    FingerprintingDataError,
    build_fingerprinting_dataset,
)


def _feature_file_identity(path: Path):
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            first = next(reader, None)
            if first is None:
                return None, False
            experiment_id = str(
                first.get("experiment_id", "")
            ).strip()
            has_client = "client_capture_id" in (reader.fieldnames or [])
            return experiment_id or None, has_client
    except Exception:
        return None, False


def discover_inputs(root: Path):
    candidates = sorted(
        path
        for path in root.rglob("*_features.csv")
        if (
            "fingerprinting_dataset" not in path.parts
            and "_X_proxy" not in path.name
        )
    )

    # Group by the experiment_id stored inside the CSV rather than deriving it
    # from filenames. Per-client files use names such as
    # EXP__client_1_features.csv while retaining experiment_id=EXP.
    grouped = {}
    for path in candidates:
        experiment_id, has_client = _feature_file_identity(path)
        if not experiment_id:
            continue
        grouped.setdefault(experiment_id, []).append(
            (path, has_client)
        )

    proxy_features = []
    for experiment_id, files in sorted(grouped.items()):
        per_client = [path for path, flag in files if flag]
        if per_client:
            # When per-client traces exist, the mixed multi-client combined
            # feature file must not enter the classifier dataset.
            proxy_features.extend(sorted(per_client))
        else:
            proxy_features.extend(sorted(path for path, _ in files))

    ground_truth = sorted(root.rglob("*_ground_truth.jsonl"))
    experiment_ids = set(grouped)
    matched_ground_truth = [
        path
        for path in ground_truth
        if any(
            path.name.startswith(experiment_id + "_")
            for experiment_id in experiment_ids
        )
    ]

    return proxy_features, matched_ground_truth


def main() -> None:
    root = Path(".").resolve()
    proxy_features, ground_truth = discover_inputs(root)

    if not proxy_features:
        raise SystemExit(
            "No *_features.csv proxy files were found under the current "
            "project directory. Copy proxy outputs into the project first."
        )

    if not ground_truth:
        raise SystemExit(
            "No matching *_ground_truth.jsonl client/server logs were found. "
            "Copy the matching client ground-truth logs into the project."
        )

    print("Proxy feature files:")
    for path in proxy_features:
        print(f"  {path}")

    print("\nMatching ground-truth files:")
    for path in ground_truth:
        print(f"  {path}")

    print(
        "\nPolicy: X will contain proxy-observable network features only. "
        "Client/server labels are written separately to Y. Resource telemetry "
        "is never read into X."
    )

    try:
        result = build_fingerprinting_dataset(
            proxy_feature_csvs=proxy_features,
            ground_truth_jsonls=ground_truth,
            output_dir=root / "fingerprinting_dataset",
            prefix="fingerprinting",
        )
    except FingerprintingDataError as exc:
        raise SystemExit(f"Fingerprinting dataset build failed: {exc}") from exc

    print("\nFingerprinting dataset created:")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
