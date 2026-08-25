from __future__ import annotations

from pathlib import Path
import csv
import json

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


def _read_network_registrations(ground_truth_paths):
    """
    Return {(experiment_id, local_ip): client_id} from client-side
    network_registration events.

    These records are ground-truth/grouping metadata only. They never enter X.
    """
    mapping = {}
    for path in ground_truth_paths:
        try:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if record.get("event") != "network_registration":
                        continue
                    experiment_id = str(
                        record.get("experiment_id", "")
                    ).strip()
                    local_ip = str(
                        record.get("local_ip", "")
                    ).strip()
                    client_id = str(
                        record.get("client_id", "")
                    ).strip()
                    if not (experiment_id and local_ip and client_id):
                        continue
                    key = (experiment_id, local_ip)
                    previous = mapping.get(key)
                    if previous and previous != client_id:
                        raise FingerprintingDataError(
                            "Conflicting network registration for "
                            f"{experiment_id}/{local_ip}: "
                            f"{previous!r} vs {client_id!r}"
                        )
                    mapping[key] = client_id
        except json.JSONDecodeError as exc:
            raise FingerprintingDataError(
                f"Invalid JSONL ground truth: {path}"
            ) from exc
    return mapping


def _discover_manifest_client_map(root: Path, registrations):
    """
    Resolve proxy client_capture_id to the actual FL client_id.

    Proxy manifest:
        client IP -> capture alias/trace ID

    Client ground truth:
        local IP -> actual federated client_id

    The join is out-of-band and is excluded from predictors.
    """
    resolved = {}
    diagnostics = []

    for manifest_path in sorted(root.rglob("*_manifest.json")):
        try:
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except Exception:
            continue

        experiment_id = str(
            manifest.get("experiment_id", "")
        ).strip()
        aliases = (
            manifest.get("capture_isolation", {})
            .get("client_aliases", {})
            or {}
        )
        if not experiment_id or not isinstance(aliases, dict):
            continue

        for client_ip, capture_id in aliases.items():
            client_ip = str(client_ip).strip()
            capture_id = str(capture_id).strip()
            actual_client_id = registrations.get(
                (experiment_id, client_ip)
            )
            if not (
                client_ip
                and capture_id
                and actual_client_id
            ):
                continue

            key = (experiment_id, capture_id)
            previous = resolved.get(key)
            if previous and previous != actual_client_id:
                raise FingerprintingDataError(
                    "Conflicting proxy/client mapping for "
                    f"{experiment_id}/{capture_id}: "
                    f"{previous!r} vs {actual_client_id!r}"
                )
            resolved[key] = actual_client_id
            diagnostics.append(
                (
                    experiment_id,
                    client_ip,
                    capture_id,
                    actual_client_id,
                )
            )

    return resolved, diagnostics


def discover_inputs(root: Path):
    candidates = sorted(
        path
        for path in root.rglob("*_features.csv")
        if (
            "fingerprinting_dataset" not in path.parts
            and "_X_proxy" not in path.name
        )
    )

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

    registrations = _read_network_registrations(
        matched_ground_truth
    )
    client_map, diagnostics = _discover_manifest_client_map(
        root,
        registrations,
    )

    return (
        proxy_features,
        matched_ground_truth,
        client_map,
        diagnostics,
    )


def main() -> None:
    root = Path(".").resolve()
    (
        proxy_features,
        ground_truth,
        client_map,
        diagnostics,
    ) = discover_inputs(root)

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

    if diagnostics:
        print("\nAutomatically resolved proxy traces:")
        for (
            experiment_id,
            client_ip,
            capture_id,
            client_id,
        ) in diagnostics:
            correction = (
                " [corrected]"
                if capture_id != client_id
                else ""
            )
            print(
                f"  {experiment_id}: {client_ip} "
                f"{capture_id} -> {client_id}{correction}"
            )
    else:
        print(
            "\nNo IP-based client mapping was resolved. "
            "Older runs will fall back to exact "
            "client_capture_id == federated client_id matching."
        )

    print(
        "\nPolicy: X contains proxy-observable network features only. "
        "Client/server labels, IP-to-client mappings, global alignment "
        "metadata, and resource telemetry are excluded from predictors."
    )

    try:
        result = build_fingerprinting_dataset(
            proxy_feature_csvs=proxy_features,
            ground_truth_jsonls=ground_truth,
            output_dir=root / "fingerprinting_dataset",
            prefix="fingerprinting",
            client_capture_id_map=client_map,
        )
    except FingerprintingDataError as exc:
        raise SystemExit(
            f"Fingerprinting dataset build failed: {exc}"
        ) from exc

    print("\nFingerprinting dataset created:")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
