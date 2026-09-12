from __future__ import annotations

from pathlib import Path
import csv
import json

from ai_fingerprint.fingerprinting_dataset import (
    FingerprintingDataError,
    build_fingerprinting_dataset,
    canonical_experiment_id_for_feature_path,
)


def _feature_file_identity(path: Path):
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            first = next(reader, None)
            if first is None:
                return None, False
            embedded_experiment_id = str(
                first.get("experiment_id", "")
            ).strip()
            experiment_id = canonical_experiment_id_for_feature_path(
                path,
                embedded_experiment_id,
            )
            has_client = "client_capture_id" in (reader.fieldnames or [])
            return experiment_id or None, has_client
    except Exception:
        return None, False


def _ground_truth_join_id(path: Path):
    """Return neutral run_id when present, otherwise the human experiment_id."""
    run_ids = set()
    experiment_ids = set()
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                run_id = str(record.get("run_id") or "").strip()
                experiment_id = str(record.get("experiment_id", "")).strip()
                if run_id:
                    run_ids.add(run_id)
                if experiment_id:
                    experiment_ids.add(experiment_id)
    except (OSError, json.JSONDecodeError):
        return None
    if len(run_ids) == 1:
        return next(iter(run_ids))
    if len(run_ids) > 1:
        raise FingerprintingDataError(
            f"Ground-truth file {path} contains multiple run_id values: {sorted(run_ids)}"
        )
    if len(experiment_ids) == 1:
        return next(iter(experiment_ids))
    return None


def _read_network_registrations(ground_truth_paths):
    """
    Return {(experiment_id, local_ip, local_port): client_id} from client-side
    network_registration events. IP+port avoids merging stale/retry TCP sessions.

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
                    if record.get("event") not in {
                        "network_registration",
                        "network_registration_confirmed",
                    }:
                        continue
                    experiment_id = str(
                        record.get("run_id") or record.get("experiment_id", "")
                    ).strip()
                    local_ip = str(
                        record.get("local_ip", "")
                    ).strip()
                    client_id = str(
                        record.get("client_id", "")
                    ).strip()
                    try:
                        local_port = int(record.get("local_port", 0) or 0)
                    except (TypeError, ValueError):
                        local_port = 0
                    if not (experiment_id and local_ip and local_port and client_id):
                        continue
                    key = (experiment_id, local_ip, local_port)
                    previous = mapping.get(key)
                    if previous and previous != client_id:
                        raise FingerprintingDataError(
                            "Conflicting network registration for "
                            f"{experiment_id}/{local_ip}:{local_port}: "
                            f"{previous!r} vs {client_id!r}"
                        )
                    mapping[key] = client_id
        except json.JSONDecodeError as exc:
            raise FingerprintingDataError(
                f"Invalid JSONL ground truth: {path}"
            ) from exc
    return mapping


def _discover_manifest_client_map(root: Path, registrations):
    """Resolve connection-granular proxy traces to actual FL client IDs."""
    resolved = {}
    diagnostics = []
    for manifest_path in sorted(root.rglob("*_manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        experiment_id = str(manifest.get("experiment_id", "")).strip()
        per_client = manifest.get("outputs", {}).get("per_client", {}) or {}
        if experiment_id and isinstance(per_client, dict):
            for capture_id, item in per_client.items():
                if not isinstance(item, dict):
                    continue
                ip = str(item.get("client_ip", "")).strip()
                try:
                    port = int(item.get("client_port", 0) or 0)
                except (TypeError, ValueError):
                    port = 0
                actual = registrations.get((experiment_id, ip, port))
                if not actual:
                    continue
                key=(experiment_id, str(capture_id))
                previous=resolved.get(key)
                if previous and previous != actual:
                    raise FingerprintingDataError(
                        f"Conflicting proxy/client mapping for {experiment_id}/{capture_id}: {previous!r} vs {actual!r}"
                    )
                resolved[key]=actual
                diagnostics.append((experiment_id, f"{ip}:{port}", str(capture_id), actual))
        # Backward compatibility for old IP-only manifests.
        aliases = manifest.get("capture_isolation", {}).get("client_aliases", {}) or {}
        if experiment_id and isinstance(aliases, dict):
            for ip, capture_id in aliases.items():
                matches=[cid for (exp, rip, _port), cid in registrations.items() if exp==experiment_id and rip==str(ip)]
                if len(set(matches)) != 1:
                    continue
                actual=matches[0]
                resolved.setdefault((experiment_id, str(capture_id)), actual)
    return resolved, diagnostics



def _prune_tiny_retry_client_map(
    root: Path,
    client_map,
    diagnostics,
):
    """
    Remove only negligible retry/stale TCP aliases after client identity
    resolution.

    If multiple capture aliases resolve to the same federated client in one
    run, an alias is treated as a retry fragment only when BOTH hold:

      packet_count < 100
      packet_count < 0.0001 * largest alias for that run/client

    This preserves substantive reconnects while excluding tiny connection
    fragments from classifier input.
    """
    packet_counts = {}

    for manifest_path in sorted(root.rglob("*_manifest.json")):
        # Use only chunk manifests so packet counts are not double-counted
        # through collection/root summary manifests.
        if not manifest_path.parent.name.startswith("chunk_"):
            continue

        try:
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except Exception:
            continue

        embedded_experiment_id = str(
            manifest.get("experiment_id", "")
        ).strip()

        experiment_id = canonical_experiment_id_for_feature_path(
            manifest_path,
            embedded_experiment_id,
        )

        per_client = (
            (manifest.get("outputs") or {})
            .get("per_client")
            or {}
        )

        if not isinstance(per_client, dict):
            continue

        for capture_id, item in per_client.items():
            if not isinstance(item, dict):
                continue

            try:
                count = int(
                    item.get("packet_count", 0) or 0
                )
            except (TypeError, ValueError):
                count = 0

            key = (
                str(experiment_id),
                str(capture_id),
            )

            packet_counts[key] = (
                packet_counts.get(key, 0)
                + max(count, 0)
            )

    # Group aliases that currently resolve to the same real client.
    groups = {}

    for (experiment_id, capture_id), client_id in client_map.items():
        groups.setdefault(
            (str(experiment_id), str(client_id)),
            []
        ).append(str(capture_id))

    drop = set()
    reasons = []

    for (experiment_id, client_id), capture_ids in groups.items():
        counts = {
            capture_id: packet_counts.get(
                (experiment_id, capture_id),
                0,
            )
            for capture_id in capture_ids
        }

        largest = max(counts.values(), default=0)

        if largest <= 0:
            continue

        for capture_id, count in counts.items():
            if (
                count < 100
                and count < largest * 0.0001
            ):
                key = (
                    experiment_id,
                    capture_id,
                )

                drop.add(key)

                reasons.append(
                    (
                        experiment_id,
                        capture_id,
                        client_id,
                        count,
                        largest,
                    )
                )

    if not drop:
        return client_map, diagnostics

    pruned = dict(client_map)

    for key in drop:
        pruned.pop(key, None)

    # Keep diagnostics consistent with the pruned mapping.
    kept_diagnostics = []

    for item in diagnostics:
        try:
            key = (
                str(item[0]),
                str(item[2]),
            )
        except Exception:
            kept_diagnostics.append(item)
            continue

        if key not in drop:
            kept_diagnostics.append(item)

    print(
        "\nTiny retry proxy traces excluded from classifier input:"
    )

    for (
        experiment_id,
        capture_id,
        client_id,
        count,
        largest,
    ) in sorted(reasons):
        print(
            f"  {experiment_id}: "
            f"{capture_id} -> {client_id} "
            f"packets={count} "
            f"largest_same_client={largest}"
        )

    return pruned, kept_diagnostics



def _chunk_manifest_backs_feature(
    path: Path,
    experiment_id: str,
    capture_id: str,
) -> bool:
    """
    Return True when a per-client chunk feature is backed by a
    same-directory finalized chunk manifest.

    This rule is applied only to:
        .../capture_artifacts/chunk_*/..._features.csv

    Historical/non-chunk layouts retain their previous behavior.

    A chunk feature is considered manifest-backed only when at least one
    same-directory *_manifest.json belongs to the canonical coordinated run
    and contains the capture alias in outputs.per_client.
    """
    path = Path(path)

    if (
        path.parent.name.startswith("chunk_")
        and path.parent.parent.name == "capture_artifacts"
    ):
        manifests = sorted(
            path.parent.glob("*_manifest.json")
        )

        if not manifests:
            return False

        for manifest_path in manifests:
            try:
                manifest = json.loads(
                    manifest_path.read_text(
                        encoding="utf-8"
                    )
                )
            except Exception:
                continue

            embedded_run_id = str(
                manifest.get("run_id")
                or manifest.get("experiment_id")
                or ""
            ).strip()

            manifest_run_id = (
                canonical_experiment_id_for_feature_path(
                    manifest_path,
                    embedded_run_id,
                )
            )

            if str(manifest_run_id) != str(experiment_id):
                continue

            per_client = (
                (manifest.get("outputs") or {})
                .get("per_client")
                or {}
            )

            if (
                isinstance(per_client, dict)
                and str(capture_id) in per_client
            ):
                return True

        return False

    return True


def discover_inputs(root: Path, allowed_experiment_ids=None):
    candidates = sorted(
        path
        for path in root.rglob("*_features.csv")
        if (
            "fingerprinting_dataset" not in path.parts
            and "_X_proxy" not in path.name
            # Progressive live-monitor features are a separate
            # inference stream. Offline dataset construction uses
            # finalized post-capture per-client features so live
            # retries/reconnections are not mixed with canonical
            # capture artifacts.
            and not path.name.endswith("_live_features.csv")
        )
    )

    allowed = (
        {str(value) for value in allowed_experiment_ids}
        if allowed_experiment_ids is not None
        else None
    )
    grouped = {}
    excluded_unmanifested = []

    for path in candidates:
        experiment_id, has_client = _feature_file_identity(path)

        if not experiment_id:
            continue

        if allowed is not None and experiment_id not in allowed:
            continue

        if has_client:
            capture_id = ""

            try:
                with path.open(
                    newline="",
                    encoding="utf-8",
                ) as handle:
                    first = next(
                        csv.DictReader(handle),
                        None,
                    )

                capture_id = str(
                    (first or {}).get(
                        "client_capture_id",
                        "",
                    )
                ).strip()

            except Exception:
                capture_id = ""

            if (
                capture_id
                and not _chunk_manifest_backs_feature(
                    path,
                    experiment_id,
                    capture_id,
                )
            ):
                excluded_unmanifested.append(path)
                continue

        grouped.setdefault(experiment_id, []).append(
            (path, has_client)
        )

    if excluded_unmanifested:
        print(
            "\nUnmanifested/incomplete chunk features "
            "excluded from classifier input:"
        )

        for path in excluded_unmanifested:
            print(f"  {path}")

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
        if _ground_truth_join_id(path) in experiment_ids
    ]

    registrations = _read_network_registrations(
        matched_ground_truth
    )
    client_map, diagnostics = _discover_manifest_client_map(
        root,
        registrations,
    )

    client_map, diagnostics = _prune_tiny_retry_client_map(
        root,
        client_map,
        diagnostics,
    )

    # In connection-granular captures, a stale/retry TCP connection from the
    # same IP may have its own trace. If at least one trace for a run maps to a
    # confirmed client registration, only confirmed traces enter classifier X.
    mapped_experiments = {experiment_id for experiment_id, _ in client_map}
    filtered_features = []
    excluded_unmatched = []
    for feature_path in proxy_features:
        experiment_id, has_client = _feature_file_identity(feature_path)
        if not experiment_id or not has_client or experiment_id not in mapped_experiments:
            filtered_features.append(feature_path)
            continue
        capture_id = ""
        try:
            with feature_path.open(newline="", encoding="utf-8") as handle:
                first = next(csv.DictReader(handle), None)
                capture_id = str((first or {}).get("client_capture_id", "")).strip()
        except Exception:
            pass
        if capture_id and (experiment_id, capture_id) in client_map:
            filtered_features.append(feature_path)
        else:
            excluded_unmatched.append(feature_path)

    if excluded_unmatched:
        print("\nUnmatched proxy connections excluded from classifier input:")
        for path in excluded_unmatched:
            print(f"  {path}")
    proxy_features = filtered_features

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
