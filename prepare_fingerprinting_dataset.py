from __future__ import annotations

from pathlib import Path

from ai_fingerprint.fingerprinting_dataset import (
    FingerprintingDataError,
    build_fingerprinting_dataset,
)


def discover_inputs(root: Path):
    proxy_features = sorted(
        path
        for path in root.rglob("*_features.csv")
        if (
            "fingerprinting_dataset" not in path.parts
            and "_X_proxy" not in path.name
        )
    )
    ground_truth = sorted(
        root.rglob("*_ground_truth.jsonl")
    )

    experiment_ids = {
        path.name[: -len("_features.csv")]
        for path in proxy_features
    }

    matched_ground_truth = [
        path
        for path in ground_truth
        if any(
            path.name.startswith(
                experiment_id + "_"
            )
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
